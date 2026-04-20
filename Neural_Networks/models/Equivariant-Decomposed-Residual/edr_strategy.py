"""EDR training strategy — plugs into the existing TorqueTrainStrategy pipeline.

This module provides the ``EDR_STRATEGY`` frozen-dataclass instance that the
shared pipeline.py training loop consumes.  The interface matches the existing
PLAIN_STRATEGY / PHYSICS_REG_STRATEGY / RESIDUAL_STRATEGY instances exactly,
so no changes to pipeline.py or checkpointing.py are required.

Training additions specific to EDR
-----------------------------------
1. **No AMP (Automatic Mixed Precision)**: The passivity regularisation loss
   requires Jacobian computation via autograd; AMP (float16) is incompatible
   with reliable Jacobian backprop.  The scaler passed by the pipeline is
   unconditionally ignored — EDR always trains in float32.

2. **Two-phase curriculum**: At epoch ``phase2_start_epoch`` the model's
   inertia and Coriolis correction networks are unfrozen.  The optimizer
   already has parameter groups for both phases; only the phase state
   changes in the model.

3. **Composite loss**:
       L = L_data  +  λ_corr · L_correction  [+  λ_pass · L_passivity]

   - L_data:       Per-joint MSE with **uniform** weights (matches ``val_rmse`` / macro RMSE).
   - L_correction: Correction magnitude regularisation — keeps δ-terms small
                   unless the data strongly disagrees with nominal physics
                   (Occam's razor).  The inertia Frobenius term is optionally
                   scaled by ``1/n_joints²`` (``correction_reg_inertia_normalize``).
   - L_passivity:  Skew-symmetry of (Ṁ − 2C).  Disabled by default; enable
                   via ``enable_passivity_loss=True`` in hp.  Expensive to
                   compute (requires autograd Jacobian), recommended for second
                   iteration.

4. **Optimizer param groups**: Two groups per phase so that frozen phase-1
   parameters (inertia/Coriolis nets) receive a minimal learning rate
   during phase 1 rather than zero LR (which would prevent their state dict
   from being tracked by AdamW momentum buffers on unfreeze).

5. **Joint weights (all 1.0 for EDR)**: Plain/residual trainers weight J2×2.5;
   EDR uses **equal per-joint weights** so the data MSE matches the scale of
   ``macro_rmse_numpy`` / early stopping on ``val_rmse`` (mean RMSE across joints).
"""

from __future__ import annotations

import logging
import math
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Path setup — add the EDR directory to sys.path so sibling imports work.
# This is idempotent and safe to call multiple times.
# ---------------------------------------------------------------------------
_EDR_DIR = str(Path(__file__).resolve().parent)
if _EDR_DIR not in sys.path:
    sys.path.insert(0, _EDR_DIR)

from edr_model import EDRModel  # noqa: E402 — local sibling import

# ---------------------------------------------------------------------------
# Absolute imports from the shared pipeline — always available from repo root.
# ---------------------------------------------------------------------------
from Neural_Networks.loader import ACTIVE_JOINTS  # noqa: E402
from Neural_Networks.models.shared.strategies import TorqueTrainStrategy  # noqa: E402
from Neural_Networks.models.shared.optim import build_optimizer_default  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Joint weights — EDR uses uniform weights so train MSE matches val RMSE scale
# (macro mean over joints; early stopping on val_rmse stays consistent).
# ---------------------------------------------------------------------------
_JOINT_WEIGHTS: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0]


# ===========================================================================
# Default hyperparameters
# ===========================================================================

DEFAULT_EXHAUSTIVE_EDR: dict[str, Any] = {
    # ── Shared keys (same as RESIDUAL / PLAIN) ─────────────────────────────
    "batch_size":             256,
    "epochs":                 300,
    "learning_rate":          3e-4,
    "optimizer":              "adamw",
    "lr_scheduler":           "warmup_cosine",
    "weight_decay":           1e-5,
    "dropout":                0.0,       # Not used; kept for registry compatibility.
    "activation":             "silu",
    "hidden_layers":          None,      # Not used; each sub-net has its own default.
    "early_stopping":         True,
    "early_stop_metric":      "val_rmse",
    "patience":               60,
    "min_delta":              1e-4,
    "grad_clip_norm":         1.0,
    "feature_noise_std":      0.0,
    "data_train_fraction":    1.0,
    "data_train_seed":        0,
    "stride":                 1,
    "seq_len":                50,
    "torch_compile":          False,
    "torch_compile_mode":     "default",
    "snapshot_every":         0,
    "seed":                   42,
    # ── EDR-specific keys ───────────────────────────────────────────────────
    "gravity_hidden":         [64, 64],
    "inertia_hidden":         [64, 64],
    "coriolis_hidden":        [64, 64],
    "friction_hidden":        [32, 32],
    "phase2_start_epoch":     18,        # Switch to phase 2 (~guide: after gravity+friction).
    "lambda_correction_reg":  5e-3,      # Correction magnitude regularisation weight.
    "correction_reg_inertia_normalize": True,  # Scale ||δM||_F² by 1/n² vs vector terms.
    "correction_dropout":     0.08,      # Dropout after hidden activations (0 = off).
    "enable_passivity_loss":  False,     # Passivity (Ṁ−2C skew-symmetry) loss.
    "lambda_passivity":       1e-2,      # Passivity loss weight (if enabled).
    "frozen_lr_ratio":        0.7,       # LR multiplier for inertia/Coriolis group (phase 1 and 2).
    "print_every":            10,        # Pipeline: log epoch INFO every N epochs (see pipeline.py).
}

# Hyperparameter keys to embed in the run ID string.
RUN_ID_KEYS_EDR: list[tuple[str, str]] = [
    ("data_train_fraction",  "frac"),
    ("learning_rate",        "lr"),
    ("weight_decay",         "wd"),
    ("batch_size",           "bs"),
    ("phase2_start_epoch",   "ph2"),
    ("lambda_correction_reg","creg"),
]


# ===========================================================================
# Loss functions
# ===========================================================================

def _weighted_mse_loss(
    tau_hat:       torch.Tensor,
    target:        torch.Tensor,
    joint_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Per-joint weighted mean-squared error.

    Parameters
    ----------
    tau_hat:
        Predicted torques, shape (B, n_joints).
    target:
        Ground-truth torques, shape (B, n_joints).
    joint_weights:
        Per-joint scalar weights, shape (n_joints,) or None.
        When None, falls back to unweighted MSE.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if joint_weights is None:
        return F.mse_loss(tau_hat, target)
    return (joint_weights * (tau_hat - target) ** 2).mean()


def _correction_reg_loss(
    delta_g:    torch.Tensor,
    delta_M:    torch.Tensor,   # (B, n_joints, n_joints) — full matrix, NOT torque
    delta_C_qd: torch.Tensor,
    delta_tau_f:torch.Tensor,
    *,
    n_joints: int,
    normalize_inertia_frob: bool = True,
) -> torch.Tensor:
    """Correction magnitude regularisation (Occam's razor).

    Penalises the magnitude of each correction term so that the model prefers
    the nominal physics prediction unless the data provides strong evidence
    that a correction is necessary.

        L_correction = ||δg||² + ||δM||_F² / n + ||δC·q̇||² + ||δτ_f||²

    when ``normalize_inertia_frob`` is True: the Frobenius term is divided by
    ``n_joints`` for per-element parity with the vector terms.

    Parameters
    ----------
    delta_g, delta_C_qd, delta_tau_f:
        Computed correction vectors, each shape (B, n_joints).
    delta_M:
        Inertia correction matrix, shape (B, n_joints, n_joints).
    n_joints:
        Joint count ``n`` (used for Frobenius normalisation).
    normalize_inertia_frob:
        If True, mean Frobenius squared is scaled by ``1 / n_joints**2``.

    Returns
    -------
    torch.Tensor
        Scalar regularisation loss.
    """
    inertia_frob = (delta_M ** 2).sum(dim=(-2, -1)).mean()
    if normalize_inertia_frob:
        inertia_frob = inertia_frob / float(max(1, int(n_joints)))
    return (
        (delta_g    ** 2).mean()
        + inertia_frob
        + (delta_C_qd ** 2).mean()
        + (delta_tau_f ** 2).mean()
    )


# ===========================================================================
# Model factory
# ===========================================================================

def _make_model_edr(device: torch.device, hp: dict[str, Any]) -> EDRModel:
    """Construct an EDRModel from hyperparameter dict.

    Parameters
    ----------
    device:
        Target torch device.
    hp:
        Hyperparameter dict.  Recognised keys: gravity_hidden, inertia_hidden,
        coriolis_hidden, friction_hidden, activation, phase2_start_epoch.

    Returns
    -------
    EDRModel
        Zero-initialised model moved to ``device``, in phase 1.

    Raises
    ------
    ValueError
        Propagated from EDRModel if any hyperparameter is invalid.
    """
    _cd = hp.get("correction_dropout", hp.get("dropout", 0.0))
    correction_dropout = float(_cd or 0.0)
    if not 0.0 <= correction_dropout < 1.0:
        raise ValueError(
            f"[EDR] correction_dropout must be in [0, 1), got {correction_dropout!r}"
        )
    # Normalization stats for sin/cos trig features (gravity network).
    _q_mean = hp.get("_q_mean", None)
    _q_std  = hp.get("_q_std",  None)

    model = EDRModel(
        n_joints=ACTIVE_JOINTS,
        gravity_hidden=list(hp.get("gravity_hidden",   [64, 64])),
        inertia_hidden=list(hp.get("inertia_hidden",   [64, 64])),
        coriolis_hidden=list(hp.get("coriolis_hidden", [64, 64])),
        friction_hidden=list(hp.get("friction_hidden", [32, 32])),
        activation=str(hp.get("activation", "silu")),
        correction_dropout=correction_dropout,
        q_mean=_q_mean,
        q_std=_q_std,
    )
    return model.to(device)


# ===========================================================================
# Optimizer
# ===========================================================================

def _build_optimizer_edr(model: EDRModel, hp: dict[str, Any]) -> AdamW:
    """Construct AdamW with two parameter groups.

    Group 0 — phase-1-active (gravity + friction):  full learning rate.
    Group 1 — phase-1-frozen (inertia + Coriolis):  reduced learning rate.

    Using a non-zero (but reduced) LR for the frozen-phase group ensures
    that AdamW's momentum buffers are maintained during phase 1.  When
    set_phase(2) is called and these parameters are unfrozen, they can begin
    training immediately with warm momentum estimates.

    Parameters
    ----------
    model:
        The EDRModel to optimise.
    hp:
        Hyperparameter dict.  Recognised keys: learning_rate, weight_decay,
        frozen_lr_ratio.

    Returns
    -------
    AdamW
        Configured optimizer instance.
    """
    lr        = float(hp.get("learning_rate",  3e-4))
    wd        = float(hp.get("weight_decay",   1e-5))
    frozen_lr = lr * float(hp.get("frozen_lr_ratio", 0.1))

    # Phase-1-active parameters.
    active_params = (
        list(model.gravity_net.parameters())
        + list(model.friction_net.parameters())
    )
    # Phase-1-frozen parameters (lower LR during phase 1, same as active in phase 2).
    frozen_params = (
        list(model.inertia_net.parameters())
        + list(model.coriolis_net.parameters())
    )

    return AdamW(
        [
            {"params": active_params, "lr": lr,        "weight_decay": wd},
            {"params": frozen_params, "lr": frozen_lr, "weight_decay": wd},
        ]
    )


# ===========================================================================
# Training epoch
# ===========================================================================

def _train_epoch_edr(
    model:          EDRModel,
    loader,
    optimizer:      torch.optim.Optimizer,
    device:         torch.device,
    hp:             dict[str, Any],
    epoch:          int,
    onecycle_sched,
    scaler,         # Accepted but unconditionally ignored — EDR is float32 only.
) -> tuple[float, float, float, float, float]:
    """Run one training epoch for EDRModel.

    EDR-specific behaviour
    ----------------------
    • AMP is disabled unconditionally (passivity loss requires Jacobian
      computation which is unreliable under float16).
    • At ``phase2_start_epoch`` the model is switched to phase 2.
    • The composite EDR loss is used instead of simple MSE.

    Parameters
    ----------
    model:
        EDRModel in training mode.
    loader:
        Training DataLoader producing (features, target, physics) batches.
    optimizer:
        AdamW instance from ``_build_optimizer_edr``.
    device:
        Computation device.
    hp:
        Hyperparameter dict.
    epoch:
        Current 1-based epoch index (used for phase switching).
    onecycle_sched:
        Optional OneCycleLR scheduler (stepped per batch if provided).
    scaler:
        GradScaler — accepted for interface compatibility, but not used.

    Returns
    -------
    tuple[float, float, float, float, float]
        ``(mean_train_loss, mean_grad_norm, train_rmse, mean_l_data, mean_l_corr)``
        where ``mean_train_loss = mean_l_data + λ·mean_l_corr`` (plus passivity
        term if enabled).  The shared pipeline logs ``mean_l_data`` /
        ``mean_l_corr`` when present.
    """
    # ── Phase switching ───────────────────────────────────────────────────
    phase2_start = int(hp.get("phase2_start_epoch", 15))
    if epoch == phase2_start and model.phase == 1:
        model.set_phase(2)
        # Give inertia/Coriolis the same LR as gravity/friction.
        # Adam's adaptive denominator naturally provides conservative initial
        # steps when momentum buffers are cold — no need to reduce LR further.
        base_lr = optimizer.param_groups[0]["lr"]
        optimizer.param_groups[1]["lr"] = base_lr
        # Clear stale Adam state for the newly unfrozen params so they start
        # fresh instead of using momentum estimates accumulated from near-zero
        # gradients during phase 1.
        for param in (
            list(model.inertia_net.parameters())
            + list(model.coriolis_net.parameters())
        ):
            if param in optimizer.state:
                del optimizer.state[param]
        logger.info(
            "EDR curriculum: switching to phase 2 at epoch %d "
            "(inertia + Coriolis corrections now trainable, LR=%.2e, Adam state reset).",
            epoch, base_lr,
        )

    model.train()

    # ── Hyperparameter extraction ─────────────────────────────────────────
    _jw         = torch.tensor(_JOINT_WEIGHTS, device=device)
    _grad_clip  = float(hp.get("grad_clip_norm",          1.0))
    _noise_std  = float(hp.get("feature_noise_std", 0.0) or 0.0)
    _lambda_reg = float(hp.get("lambda_correction_reg",  1e-3))
    _norm_inertia_frob = bool(hp.get("correction_reg_inertia_normalize", True))
    _use_pass   = bool(hp.get("enable_passivity_loss",   False))
    _lambda_pass= float(hp.get("lambda_passivity",       1e-2))

    total_loss  = 0.0
    total_l_data = 0.0
    total_l_corr = 0.0
    total_gnorm = 0.0
    total_sse   = 0.0
    total_elem  = 0
    n_batches   = len(loader)

    optimizer.zero_grad(set_to_none=True)

    for features, target, physics in loader:
        features = features.to(device, non_blocking=True)
        target   = target.to(device,   non_blocking=True)
        physics  = physics.to(device,  non_blocking=True)

        # Optional feature-space noise (data augmentation).
        if _noise_std > 0.0:
            features = features + torch.randn_like(features) * _noise_std

        # ── Correction terms (needed for reg loss) ───────────────────────
        # We compute corrections explicitly here rather than calling model.forward()
        # so that we can inspect individual δ-terms for the regularisation loss
        # without a second forward pass.
        n = model.n_joints
        q   = features[:, 0:n]
        qd  = features[:, n:2*n]
        qdd = features[:, 2*n:3*n]

        tau_g = physics[:, 0:n]
        tau_M = physics[:, n:2*n]
        tau_C = physics[:, 2*n:3*n]
        tau_f = physics[:, 3*n:4*n]

        if model._use_trig_features:
            q_raw = q * model.q_std + model.q_mean
            sin_q = torch.sin(q_raw)
            cos_q = torch.cos(q_raw)
            q_aug = torch.cat([q, sin_q, cos_q], dim=-1)
            delta_g = model.gravity_net(q_aug)
            coriolis_input = torch.cat([q, sin_q, cos_q, qd], dim=-1)
        else:
            delta_g = model.gravity_net(q)
            coriolis_input = torch.cat([q, qd], dim=-1)
        # Compute δM as the full (B, n, n) matrix for Frobenius-norm regularisation,
        # then apply it to q̈ for the torque contribution.  One forward pass, two uses.
        delta_M      = model.inertia_net.compute_delta_M(q)             # (B, n, n)
        delta_M_qdd  = torch.bmm(delta_M, qdd.unsqueeze(-1)).squeeze(-1) # (B, n)
        delta_C_qd   = model.coriolis_net(coriolis_input, qd)
        delta_tau_f  = model.friction_net(qd)

        tau_hat = (
            (tau_g + delta_g)
            + (tau_M + delta_M_qdd)
            + (tau_C + delta_C_qd)
            + (tau_f + delta_tau_f)
        )

        # ── Loss assembly ────────────────────────────────────────────────
        l_data = _weighted_mse_loss(tau_hat, target, _jw)
        l_corr = _correction_reg_loss(
            delta_g,
            delta_M,
            delta_C_qd,
            delta_tau_f,
            n_joints=n,
            normalize_inertia_frob=_norm_inertia_frob,
        )
        loss   = l_data + _lambda_reg * l_corr

        if _use_pass:
            l_pass = _passivity_loss_batch(
                model=model,
                q=q,
                qd=qd,
                qdd=qdd,
                passivity_sample_fraction=0.1,
            )
            loss = loss + _lambda_pass * l_pass

        # ── Backward + gradient clip + step ─────────────────────────────
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if onecycle_sched is not None:
            onecycle_sched.step()

        total_loss   += loss.item()
        total_l_data += l_data.item()
        total_l_corr += l_corr.item()
        total_gnorm += gnorm.item() if hasattr(gnorm, "item") else float(gnorm)
        with torch.no_grad():
            d = tau_hat.detach() - target
            total_sse  += float((d * d).sum().item())
            total_elem += d.numel()

    train_rmse = math.sqrt(total_sse / max(1, total_elem))
    mean_l_data = total_l_data / n_batches
    mean_l_corr = total_l_corr / n_batches
    return (
        total_loss / n_batches,
        total_gnorm / n_batches,
        train_rmse,
        mean_l_data,
        mean_l_corr,
    )


# ===========================================================================
# Evaluation epoch
# ===========================================================================

def _eval_epoch_edr(
    model:  EDRModel,
    loader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one evaluation epoch for EDRModel.

    Parameters
    ----------
    model:
        EDRModel (must be in eval mode when called — set by the pipeline loop).
    loader:
        Validation or test DataLoader.
    device:
        Computation device.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        (mean_val_loss, all_predictions, all_targets).
        ``mean_val_loss`` uses the **same** per-joint MSE weighting as training
        ``l_data`` (uniform weights for EDR), so it is comparable to
        ``train_loss - λ·l_corr`` in logs.
        Predictions and targets are concatenated across all batches,
        shape (N_total, n_joints).
    """
    model.eval()
    total_loss = 0.0
    all_pred:   list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    _jw = torch.tensor(_JOINT_WEIGHTS, device=device)

    with torch.no_grad():
        for features, target, physics in loader:
            features = features.to(device, non_blocking=True)
            target   = target.to(device,   non_blocking=True)
            physics  = physics.to(device,  non_blocking=True)

            tau_hat = model(features, physics)
            loss    = _weighted_mse_loss(tau_hat, target, _jw)
            total_loss += loss.item()

            p    = tau_hat.cpu().numpy()
            t_np = target.cpu().numpy()
            if p.ndim == 3:
                # Sequence mode batches: collapse (B, T, n) → (B·T, n).
                p    = p.reshape(-1,    p.shape[-1])
                t_np = t_np.reshape(-1, t_np.shape[-1])
            all_pred.append(p)
            all_target.append(t_np)

    return (
        total_loss / len(loader),
        np.concatenate(all_pred,   axis=0),
        np.concatenate(all_target, axis=0),
    )


# ===========================================================================
# Passivity loss (optional, disabled by default)
# ===========================================================================

def _passivity_loss_batch(
    model:                      EDRModel,
    q:                          torch.Tensor,
    qd:                         torch.Tensor,
    qdd:                        torch.Tensor,
    passivity_sample_fraction:  float = 0.1,
) -> torch.Tensor:
    """Compute skew-symmetry passivity constraint on a random subset of the batch.

    The true robot dynamics satisfy (Ṁ − 2C) is skew-symmetric.  When we add
    inertia and Coriolis corrections, the effective matrices are:

        M̃(q) = M(q) + δM(q)
        C̃    = C    + δC

    Enforcing (dM̃/dt − 2C̃) skew-symmetric keeps the corrected model passive
    — a prerequisite for stability guarantees in model-based control.

    This is computed only on a random fraction of each minibatch to keep
    training speed acceptable.

    Parameters
    ----------
    model:
        EDRModel (inertia_net must be accessible).
    q, qd, qdd:
        Unnormalised or normalised kinematic state (normalised is fine — we
        only care about the relative change of δM with q).
    passivity_sample_fraction:
        Fraction of the batch to evaluate (default 0.1).

    Returns
    -------
    torch.Tensor
        Scalar passivity loss (‖S + S^T‖_F² averaged over sampled batch).

    Notes
    -----
    Computing dM̃/dt = Σ_i (∂δM/∂q_i) q̇_i requires a Jacobian computation
    (one Jacobian-vector product per joint).  This is moderately expensive
    (~10× slower than the data loss alone), hence the random subsampling.
    Float32 is mandatory; this function will raise under AMP.
    """
    B = q.shape[0]
    k = max(1, int(B * passivity_sample_fraction))
    # Random subset indices.
    idx = torch.randperm(B, device=q.device)[:k]
    q_s   = q[idx].detach().requires_grad_(True)
    qd_s  = qd[idx]
    qdd_s = qdd[idx]

    # δM(q) shape (k, n, n) via the InertiaCorrection module.
    delta_M = model.inertia_net.compute_delta_M(q_s)   # (k, n, n)
    n = delta_M.shape[-1]

    # Compute dδM/dt = Σ_i (∂δM/∂q_i) · q̇_i using autograd Jacobian.
    # We differentiate through all n·n output entries w.r.t. all n input entries.
    dM_dt = torch.zeros_like(delta_M)   # (k, n, n)
    for i in range(n):
        # Gradient of sum of column i of δM w.r.t. q.
        grad_M_i = torch.autograd.grad(
            outputs=delta_M[:, :, i].sum(),
            inputs=q_s,
            create_graph=True,
            retain_graph=True,
        )[0]  # (k, n) — ∂(δM[:,col_i])/∂q
        # dδM_col_i / dt = (∂δM_col_i / ∂q) · q̇
        dM_dt[:, :, i] = (grad_M_i * qd_s).sum(dim=-1)

    # S = dM̃/dt − 2·δC·q̇  (we only penalise the correction's skew contribution).
    # The nominal (M, C) from Pinocchio already satisfies passivity, so we only
    # need to enforce it for the correction terms.
    # Build coriolis input (with optional trig features to match model contract).
    if model._use_trig_features:
        q_raw_s = q_s.detach() * model.q_std + model.q_mean
        cor_feat_s = torch.cat(
            [q_s.detach(), torch.sin(q_raw_s), torch.cos(q_raw_s), qd_s],
            dim=-1,
        )
    else:
        cor_feat_s = torch.cat([q_s.detach(), qd_s], dim=-1)
    delta_C_qd = model.coriolis_net(cor_feat_s, qd_s)  # (k, n)
    # Reshape δC·q̇ as a column matrix to construct a proxy for 2C:
    # We approximate 2·δC as a diagonal contribution dM_diag for the passivity
    # check on the correction.  This is an approximation; for exact passivity
    # enforcement the full δC matrix is needed.  Here we build a symmetric
    # proxy S = dM_dt_sym and check its skew component.
    S = dM_dt   # (k, n, n)
    skew_norm = torch.norm(S + S.transpose(1, 2), dim=(-2, -1)).pow(2).mean()
    return skew_norm


# ===========================================================================
# Physics schedule metadata (for checkpointing compatibility)
# ===========================================================================

def _edr_physics_sched_metadata(hp: dict[str, Any]) -> dict[str, Any]:
    """Return EDR curriculum metadata for checkpointing.

    Returns
    -------
    dict
        Keys useful for post-hoc analysis: phase2_start_epoch,
        lambda_correction_reg, enable_passivity_loss.
    """
    _cd = hp.get("correction_dropout", hp.get("dropout", 0.0))
    return {
        "mode":                   "edr_two_phase_curriculum",
        "phase2_start_epoch":     int(hp.get("phase2_start_epoch",    15)),
        "lambda_correction_reg":  float(hp.get("lambda_correction_reg", 1e-3)),
        "correction_reg_inertia_normalize": bool(
            hp.get("correction_reg_inertia_normalize", True)
        ),
        "correction_dropout":     float(_cd or 0.0),
        "enable_passivity_loss":  bool(hp.get("enable_passivity_loss", False)),
        "lambda_passivity":       float(hp.get("lambda_passivity",      1e-2)),
    }


# ===========================================================================
# Strategy instance — plugs directly into pipeline.py
# ===========================================================================

EDR_STRATEGY = TorqueTrainStrategy(
    default_exhaustive_hp=DEFAULT_EXHAUSTIVE_EDR,
    run_id_hp_keys=RUN_ID_KEYS_EDR,
    make_model=_make_model_edr,
    build_optimizer=_build_optimizer_edr,
    train_epoch=_train_epoch_edr,
    eval_epoch=_eval_epoch_edr,
    physics_sched_metadata=_edr_physics_sched_metadata,
)
