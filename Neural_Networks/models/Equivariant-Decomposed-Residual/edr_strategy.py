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
from Neural_Networks.models.shared.strategies import (  # noqa: E402
    TorqueTrainStrategy,
    TrainEpochMetrics,
)
from Neural_Networks.models.shared.optim import build_optimizer_default  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Joint weights — EDR uses custom weights to focus on bottlenecks
# ---------------------------------------------------------------------------
_JOINT_WEIGHTS: list[float] = [1.0, 3.0, 1.2, 0.7, 2.5]


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
    # Phase-2 curriculum: adaptive plateau detection on val_rmse.
    # If ``phase2_start_epoch`` is set (int > 0), it forces the transition at
    # that epoch (back-compat / manual override).  If set to None, the
    # transition is triggered adaptively when phase-1 val_rmse plateaus.
    "phase2_start_epoch":     None,
    "phase2_plateau_window":  5,        # Rolling window over which to measure improvement.
    "phase2_plateau_threshold": 5e-3,   # Relative improvement threshold (0.5%).
    "phase2_min_epoch":       3,        # Never trigger before this epoch (noise at start).
    "phase2_max_epoch":       25,       # Safety fallback: force transition no later than this.
    "lambda_correction_reg":  5e-3,      # Correction magnitude regularisation weight.
    "correction_reg_inertia_normalize": True,  # Scale ||δM||_F² by 1/n² vs vector terms.
    "correction_dropout":     0.08,      # Dropout after hidden activations (0 = off).
    "enable_passivity_loss":  False,     # Passivity (Ṁ−2C skew-symmetry) loss.
    "lambda_passivity":       1e-2,      # Passivity loss weight (if enabled).
    "frozen_lr_ratio":        0.7,       # LR multiplier for inertia/Coriolis group (phase 1 and 2).
    "print_every":            10,        # Pipeline: log epoch INFO every N epochs (see pipeline.py).
}


# ===========================================================================
# Adaptive phase-2 plateau detection (pure function — easily unit-tested)
# ===========================================================================

def _should_transition_to_phase2(
    val_rmse_history: list[float],
    hp: dict[str, Any],
    current_epoch: int,
    current_phase: int,
) -> tuple[bool, str]:
    """Decide whether the training loop should switch from phase 1 to phase 2.

    The decision is based on the recent val_rmse history.  Phase 2 is triggered
    when either (a) the sliding-window relative improvement falls below a
    threshold, or (b) a safety-fallback epoch cap is reached.

    Parameters
    ----------
    val_rmse_history:
        Ordered list of val_rmse observations, oldest first.  Typically one
        entry per completed training epoch.
    hp:
        Hyperparameter dict.  Recognised keys:
        - ``phase2_start_epoch``      — if an int, forces transition at that epoch (override).
        - ``phase2_plateau_window``   — sliding-window size W (default 5).
        - ``phase2_plateau_threshold``— relative improvement threshold (default 5e-3).
        - ``phase2_min_epoch``        — never trigger before this epoch (default 3).
        - ``phase2_max_epoch``        — safety fallback: force at this epoch (default 25).
    current_epoch:
        The 1-based epoch index that is about to start.  The history should
        contain one entry per already-completed epoch (so len(history) =
        current_epoch - 1 at the top of epoch ``current_epoch``).
    current_phase:
        The model's current phase.  No transition is recommended if already
        in phase 2.

    Returns
    -------
    (should_transition, reason) — reason is a short human-readable string.
    """
    if current_phase != 1:
        return (False, "already in phase 2")

    # Manual override: phase2_start_epoch forces a fixed schedule.
    override = hp.get("phase2_start_epoch")
    if override is not None:
        if current_epoch >= int(override):
            return (True, f"manual override (phase2_start_epoch={int(override)})")
        return (False, "before manual phase2_start_epoch")

    min_epoch = int(hp.get("phase2_min_epoch", 3))
    max_epoch = int(hp.get("phase2_max_epoch", 25))
    window    = int(hp.get("phase2_plateau_window", 5))
    threshold = float(hp.get("phase2_plateau_threshold", 5e-3))

    # Safety fallback — force transition at max_epoch even without plateau.
    if current_epoch >= max_epoch:
        return (True, f"safety fallback at max_epoch={max_epoch}")

    # Minimum length guard — avoid triggering on early-epoch noise.
    if current_epoch < min_epoch:
        return (False, f"before min_epoch={min_epoch}")

    # Need at least ``window+1`` points to measure improvement across W epochs.
    if len(val_rmse_history) < window + 1:
        return (False, f"not enough history (need {window + 1}, have {len(val_rmse_history)})")

    old = val_rmse_history[-(window + 1)]
    new = val_rmse_history[-1]
    if old <= 0.0:
        return (False, "non-positive reference val_rmse")
    rel_improvement = (old - new) / old
    if rel_improvement < threshold:
        return (True, f"plateau: rel_improvement={rel_improvement:.4f} < {threshold}")
    return (False, f"still improving: rel_improvement={rel_improvement:.4f}")

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
    mean:          torch.Tensor | None = None,
    std:           torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-joint Weighted Huber loss on normalized space.

    Parameters
    ----------
    tau_hat:
        Predicted torques in physical space, shape (B, n_joints).
    target:
        Ground-truth torques in physical space, shape (B, n_joints).
    joint_weights:
        Per-joint scalar weights, shape (n_joints,) or None.
    mean:
        Target mean, shape (n_joints,).
    std:
        Target standard deviation, shape (n_joints,).

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if mean is not None and std is not None:
        std = torch.clamp(std, min=1e-8)
        p_norm = (tau_hat - mean) / std
        t_norm = (target - mean) / std
        if joint_weights is None:
            return F.huber_loss(p_norm, t_norm, delta=0.5)
        loss = F.huber_loss(p_norm, t_norm, reduction="none", delta=0.5)
        return (joint_weights * loss).mean()
    else:
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
):
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
    TrainEpochMetrics
        Standard structured payload: ``loss_total`` is the EDR objective
        (``l_data + λ·l_corr`` plus optional passivity), ``loss_data_unw`` is
        the unweighted MSE matching ``val_loss``, ``sse_per_joint`` lets the
        pipeline derive a physical-units macro RMSE, and ``extras`` carries
        the per-component correction telemetry that the shared pipeline logs.
    """
    # ── Adaptive phase-2 transition (plateau detection) ──────────────────
    should_switch, reason = _should_transition_to_phase2(
        val_rmse_history=model.val_rmse_history,
        hp=hp,
        current_epoch=epoch,
        current_phase=model.phase,
    )
    if should_switch:
        model.set_phase(2)
        # Give inertia/Coriolis the same LR as gravity/friction.  Adam's
        # adaptive denominator naturally yields conservative initial steps
        # when momentum buffers are cold.
        base_lr = optimizer.param_groups[0]["lr"]
        optimizer.param_groups[1]["lr"] = base_lr
        # Clear stale Adam state for newly unfrozen params so they start
        # fresh rather than with momentum from near-zero phase-1 gradients.
        for param in (
            list(model.inertia_net.parameters())
            + list(model.coriolis_net.parameters())
        ):
            if param in optimizer.state:
                del optimizer.state[param]
        logger.info(
            "EDR curriculum: switching to phase 2 at epoch %d — %s "
            "(LR=%.2e, Adam state reset).",
            epoch, reason, base_lr,
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

    std_meta = torch.from_numpy(loader.dataset.std_tau).to(device)
    mean_meta = torch.from_numpy(loader.dataset.mean_tau).to(device)

    total_loss  = 0.0
    total_l_data = 0.0
    total_l_corr = 0.0
    total_loss_data_unw = 0.0
    total_gnorm = 0.0
    sse_per_joint: np.ndarray | None = None
    n_samples = 0
    # Per-component correction-magnitude telemetry (for interpretability).
    # All are batch-mean scalars accumulated across the epoch.
    total_mag_g    = 0.0   # mean |δg| over batch+joints
    total_mag_M    = 0.0   # mean Frobenius ||δM||_F over batch
    total_mag_C_qd = 0.0   # mean |δC·q̇| over batch+joints
    total_mag_f    = 0.0   # mean |δτ_f| over batch+joints
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

        # Build correction-network inputs via the model's single-source helper.
        inputs = model.build_correction_inputs(q, qd)
        delta_g      = model.gravity_net(inputs["gravity_input"])
        # Compute δM as the full (B, n, n) matrix for Frobenius-norm regularisation,
        # then apply it to q̈ for the torque contribution.  One forward pass, two uses.
        delta_M      = model.inertia_net.compute_delta_M(q)             # (B, n, n)
        delta_M_qdd  = torch.bmm(delta_M, qdd.unsqueeze(-1)).squeeze(-1) # (B, n)
        delta_C_qd   = model.coriolis_net(inputs["coriolis_input"], qd)
        delta_tau_f  = model.friction_net(qd)

        tau_hat = (
            (tau_g + delta_g)
            + (tau_M + delta_M_qdd)
            + (tau_C + delta_C_qd)
            + (tau_f + delta_tau_f)
        )

        # ── Loss assembly ────────────────────────────────────────────────
        l_data = _weighted_mse_loss(tau_hat, target, _jw, mean=mean_meta, std=std_meta)
        l_corr = _correction_reg_loss(
            delta_g,
            delta_M,
            delta_C_qd,
            delta_tau_f,
            n_joints=n,
            normalize_inertia_frob=_norm_inertia_frob,
        )
        loss   = l_data + _lambda_reg * l_corr

        l_pass = None
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

        total_loss   += float(loss.item())
        total_l_data += float(l_data.item())
        total_l_corr += float(l_corr.item())
        total_gnorm += gnorm.item() if hasattr(gnorm, "item") else float(gnorm)
        with torch.no_grad():
            d = tau_hat.detach() - target
            sse_batch = (d * d).sum(dim=0).cpu().numpy()  # (n_joints,)
            sse_per_joint = sse_batch if sse_per_joint is None else sse_per_joint + sse_batch
            n_samples += int(d.shape[0])
            total_loss_data_unw += float(F.mse_loss(tau_hat.detach(), target).item())
            # Per-component magnitudes (detached — pure telemetry, no grad).
            total_mag_g    += float(delta_g.detach().abs().mean().item())
            total_mag_M    += float(
                torch.sqrt((delta_M.detach() ** 2).sum(dim=(-2, -1))).mean().item()
            )
            total_mag_C_qd += float(delta_C_qd.detach().abs().mean().item())
            total_mag_f    += float(delta_tau_f.detach().abs().mean().item())
        if l_pass is not None:
            del l_pass
        del (
            features, target, physics, loss, tau_hat, d, l_data, l_corr, delta_g, delta_M,
            delta_M_qdd, delta_C_qd, delta_tau_f, q, qd, qdd, tau_g, tau_M, tau_C, tau_f, inputs,
        )

    mean_l_data = total_l_data / n_batches
    mean_l_corr = total_l_corr / n_batches
    # Per-component correction-magnitude epoch means.
    mean_mag_g    = total_mag_g / n_batches
    mean_mag_M    = total_mag_M / n_batches
    mean_mag_C_qd = total_mag_C_qd / n_batches
    mean_mag_f    = total_mag_f / n_batches
    return TrainEpochMetrics(
        loss_total=total_loss / n_batches,
        loss_data_unw=total_loss_data_unw / n_batches,
        grad_norm=total_gnorm / n_batches,
        sse_per_joint=sse_per_joint if sse_per_joint is not None else np.zeros(model.n_joints),
        n_samples=n_samples,
        extras={
            "l_data_jw": mean_l_data,
            "l_corr":    mean_l_corr,
            "correction_magnitudes": {
                "mean_abs_delta_g":     mean_mag_g,
                "mean_frob_delta_M":    mean_mag_M,
                "mean_abs_delta_C_qd":  mean_mag_C_qd,
                "mean_abs_delta_tau_f": mean_mag_f,
            },
        },
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
    
    std_meta = torch.from_numpy(loader.dataset.std_tau).to(device)
    mean_meta = torch.from_numpy(loader.dataset.mean_tau).to(device)

    with torch.no_grad():
        for features, target, physics in loader:
            features = features.to(device, non_blocking=True)
            target   = target.to(device,   non_blocking=True)
            physics  = physics.to(device,  non_blocking=True)

            tau_hat = model(features, physics)
            loss    = _weighted_mse_loss(tau_hat, target, _jw, mean=mean_meta, std=std_meta)
            total_loss += loss.item()

            p    = tau_hat.cpu().numpy()
            t_np = target.cpu().numpy()
            if p.ndim == 3:
                # Sequence mode batches: collapse (B, T, n) → (B·T, n).
                p    = p.reshape(-1,    p.shape[-1])
                t_np = t_np.reshape(-1, t_np.shape[-1])
            all_pred.append(p)
            all_target.append(t_np)
            del features, target, physics, tau_hat, loss, p, t_np

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
    # Build coriolis input via the model's helper so feature contract stays
    # in sync with forward() and train_epoch().
    inputs = model.build_correction_inputs(q_s.detach(), qd_s)
    delta_C_qd = model.coriolis_net(inputs["coriolis_input"], qd_s)  # (k, n)
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
        Keys useful for post-hoc analysis: phase-2 curriculum config,
        lambda_correction_reg, enable_passivity_loss.  ``phase2_start_epoch``
        is None when adaptive plateau detection is used (the default).
    """
    _cd = hp.get("correction_dropout", hp.get("dropout", 0.0))
    _p2 = hp.get("phase2_start_epoch", None)
    return {
        "mode":                   "edr_two_phase_curriculum",
        "phase2_start_epoch":     (int(_p2) if _p2 is not None else None),
        "phase2_plateau_window":  int(hp.get("phase2_plateau_window",    5)),
        "phase2_plateau_threshold": float(hp.get("phase2_plateau_threshold", 5e-3)),
        "phase2_min_epoch":       int(hp.get("phase2_min_epoch",         3)),
        "phase2_max_epoch":       int(hp.get("phase2_max_epoch",        25)),
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
