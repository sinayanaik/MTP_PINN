"""
Neural_Networks.core.trainer
==============================
Full model training pipeline: epoch loops, physics-weight scheduling,
early stopping, AMP, checkpoint saving and registry updates.

This module contains zero UI code.  All Rich-styled progress and tables
are emitted by the caller (``train_ui.run_interactive_train`` or ``train.py``)
using a Rich ``Console``.  The ``train_model`` function accepts an optional
``console`` keyword argument; if omitted a plain stdlib logger is used instead.

Public API
----------
PhysicsWeightScheduler         — physics loss weight (w_p) scheduler
train_epoch(...)               -> (mean_loss, mean_grad_norm, train_rmse)
eval_epoch(...)                -> (loss, all_pred, all_target)
train_model(run_dir, model_type, hp, *, paths, console) -> (save_dir, metrics)
update_registry(...)           — append run to models_registry.yaml
save_comparison_plot(...)      — save per-joint prediction plot
save_architecture_summary(...) — save model repr to .txt
"""

from __future__ import annotations

import csv
import logging
import math
import os
import random
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Neural_Networks.apps.metrics import (
    compute_metrics,
    trajectory_mean_rmse_numpy,
    macro_rmse_numpy,
)
from Neural_Networks.apps.builder import (
    build_model,
    build_optimizer,
    build_scheduler,
)
from Neural_Networks.physics import ACTIVE_JOINTS
from Neural_Networks.models import (
    MODEL_SAVE_DIRS,
    PHYSICS_INPUT_MODELS,
    PHYSICS_WEIGHT_MODELS,
)
from Neural_Networks.models.torque_fnns import _reduce_physics_to_total

logger = logging.getLogger(__name__)


# =============================================================================
# Physics weight serialisation helper
# =============================================================================

def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-JSON-serialisable objects to plain Python types.

    Handles: dict, list/tuple, Path, torch types, numpy scalars/arrays.
    """
    if isinstance(obj, dict):
        return {_make_serializable(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    if hasattr(obj, "__class__") and "torch" in type(obj).__module__:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _fmt_time(seconds: float) -> str:
    """Format seconds as a human-readable duration string (e.g. 2h 3m 45s)."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# =============================================================================
# Physics weight scheduler
# =============================================================================

class PhysicsWeightScheduler:
    """Fixed-α physics mixture-weight scheduler with a linear warmup.

    The training objective is a *convex mixture*:

        L = (1 − α) · L_data + α · L_physics,    α ∈ [0, 1]

    α is a single user-tunable hyperparameter (``physics_weight``).  All
    adaptive/curriculum/plateau logic has been removed — empirically, on
    this robot-inverse-dynamics task, adaptive schedules either amplify
    the imperfect RNEA bias (driving val_rmse up after ~10 epochs) or
    introduce sawtooth instability.  A constant α that the user picks by
    hand (e.g. 0.05, 0.10, 0.20) gives reproducible, interpretable runs.

    A brief linear warmup (α: 0 → target over ``physics_warmup_fraction``
    of total epochs) still helps prevent optimiser shock on epoch 1 when
    physics residuals are large on an uninitialised network.

    Returns
    -------
    step() -> (w_d, w_p)  with  w_d + w_p = 1  exactly.
    """

    # Legacy name kept so external code that imports MODES still works, but
    # only ``fixed`` is supported now.
    MODES = ("fixed",)

    def __init__(
        self,
        weight: float = 0.10,
        warmup_fraction: float = 0.03,
        total_epochs: int = 500,
        # Back-compat kwargs — silently accepted, ignored.
        mode: str = "fixed",
        **_legacy,
    ):
        # α ∈ [0, 1] — convex-mixture invariant.
        self._target_weight = float(max(0.0, min(1.0, float(weight))))
        self._init_weight   = self._target_weight
        self._warmup_epochs = max(0, int(warmup_fraction * total_epochs))
        self._total_epochs  = int(total_epochs)
        # Current α — starts at 0 during warmup, ramps to target.
        self.w_p            = 0.0 if self._warmup_epochs > 0 else self._target_weight
        # Back-compat: some call sites still inspect these attributes.
        self.mode           = "fixed"
        self.should_stop    = False

    def step(self, epoch: int, val_rmse: float) -> tuple[float, float]:
        """Compute (w_d, w_p) for the current epoch — pure function of ``epoch``."""
        we = self._warmup_epochs
        if we > 0 and epoch <= we:
            self.w_p = self._target_weight * (epoch / we)
        else:
            self.w_p = self._target_weight
        self.w_p = float(max(0.0, min(1.0, self.w_p)))
        return 1.0 - self.w_p, self.w_p

    def describe(self) -> str:
        """One-line human-readable description for logs / training panels."""
        warmup_tag = f"  warmup={self._warmup_epochs}ep" if self._warmup_epochs > 0 else ""
        return f"fixed  α={self._target_weight:.3f}{warmup_tag}"

    def config_dict(self) -> dict:
        """Return a serialisable config for metadata.yaml."""
        return {
            "mode":          "fixed",
            "weight":        self._init_weight,
            "warmup_epochs": self._warmup_epochs,
        }

    @classmethod
    def from_hp(cls, hp: dict) -> "PhysicsWeightScheduler":
        """Build from a flat hp dict.  Honours only ``physics_weight`` and
        ``physics_warmup_fraction``; legacy keys (``physics_sched``,
        ``curriculum_*``, ``physics_nudge_step``, ``physics_max_bad_nudges``)
        are ignored with no warning — they're simply no longer used."""
        return cls(
            weight          = float(hp.get("physics_weight", 0.10)),
            warmup_fraction = float(hp.get("physics_warmup_fraction", 0.03)),
            total_epochs    = int(hp.get("epochs", 500)),
        )


# =============================================================================
# Loss computation helpers
# =============================================================================

def _forward_pass(
    model: nn.Module,
    features: torch.Tensor,
    physics: torch.Tensor,
    model_type: str,
) -> tuple[torch.Tensor, dict]:
    """Dispatch the forward pass to the correct model signature."""
    if model_type in PHYSICS_INPUT_MODELS:
        return model(features, physics), {}
    return model(features), {}


class LossNormaliser:
    """Running-EMA rescaler so the convex-mixture loss acts on comparable magnitudes.

    Problem
    -------
    The convex mixture L = (1−α)·L_data + α·L_phys only behaves like a mixture
    when L_data and L_phys are on the same scale.  In practice they are not —
    physics residuals (RNEA discrepancy in normalised τ space) can be orders
    of magnitude larger or smaller than the data MSE depending on the model
    and how well τ_calib has converged.  Without rescaling, α becomes a
    meaningless knob: the mixture is dominated by whichever loss is larger
    in absolute value.

    Solution
    --------
    Maintain an exponential moving average of each detached loss magnitude,
    and scale L_phys by κ = μ_d / μ_p before mixing.  After ~1/(1−β) batches
    the two losses live on unit scale, so α = 0.85 genuinely means "physics
    contributes 85% of the gradient energy" — which is what the user demanded.

    The rescaling does **not** change the convex-mixture invariant
    (w_d + w_p = 1); it only fixes the semantics of α.
    """

    def __init__(self, beta: float = 0.98, eps: float = 1e-8):
        self.beta  = float(beta)
        self.eps   = float(eps)
        self.mu_d: float | None = None
        self.mu_p: float | None = None
        self._frozen           = False      # if True, stop updating (e.g. eval)

    def update(self, l_data: float, l_phys: float) -> None:
        """Update EMAs from detached floats.  Call once per training batch."""
        if self._frozen or not (math.isfinite(l_data) and math.isfinite(l_phys)):
            return
        self.mu_d = l_data if self.mu_d is None else (self.beta * self.mu_d + (1 - self.beta) * l_data)
        self.mu_p = l_phys if self.mu_p is None else (self.beta * self.mu_p + (1 - self.beta) * l_phys)

    def scale(self) -> float:
        """Return κ = μ_d / μ_p.  Falls back to 1.0 until both EMAs are primed."""
        if self.mu_d is None or self.mu_p is None or self.mu_p < self.eps:
            return 1.0
        return float(self.mu_d / self.mu_p)


def _compute_loss(
    tau_hat: torch.Tensor,
    target: torch.Tensor,
    physics: torch.Tensor,
    components: dict,
    model: nn.Module,
    model_type: str,
    loss_w_d: float,
    loss_w_p: float,
    hp: dict,
    _jw: "torch.Tensor | None" = None,
    features: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Compute the training loss for a given model type.

    Convex-mixture formulation
    --------------------------
    The data and physics *fitting* terms form a convex mixture:

        L_fit = (1 − α) · L_data + α · L_physics,    α = loss_w_p ∈ [0, 1]

    This guarantees w_data + w_physics = 1 throughout training.  Architecture
    priors (SPD / dissipative friction / correction regulariser / α-reg) are
    fixed-weight structural terms added *outside* the mixture — they enforce
    mathematical properties, not fit quality, so they are not part of the
    data↔physics trade-off.

    Per model
    ---------
    - ResidualCorrectionFNN   : L_data + λ_α·(α−1)² regulariser (no physics loss).
    - PhysicsRegularizedFNN   : (1−α)·L_data + α·L_phys      (α-mixture active)

    Parameters
    ----------
    loss_w_d : float  Data scale (1 − α).  Passed for logging; recomputed here.
    loss_w_p : float  Physics mixture weight α ∈ [0, 1].
    _jw      : joint weight tensor (5,) — J2 shoulder upweighted 2.5× to
               balance per-joint training contribution.
    """
    # Convex mixture: α clamped to [0, 1]; data side is always the complement.
    alpha  = float(max(0.0, min(1.0, loss_w_p)))
    w_data = 1.0 - alpha
    w_phys = alpha
    if _jw is None:
        _jw = torch.ones(tau_hat.shape[-1], device=tau_hat.device, dtype=tau_hat.dtype)
    # Unwrap torch.compile wrapper to access model-specific attributes
    _loss_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    # Loss normaliser: lazily attached by train_model().  When present, we use
    # it to rescale the physics fitting term so the convex mixture acts on
    # unit-scale losses (see LossNormaliser docstring).
    _norm: LossNormaliser | None = getattr(_loss_model, "_loss_normaliser", None)

    # Weighted per-joint data MSE: J2 (shoulder) carries the full arm weight
    data_loss = (_jw * (tau_hat - target) ** 2).mean()

    if model_type in {"ResidualCorrectionFNN"}:
        # Residual correction: fixed-weight regularisation on the scale factor α.
        # Not tied to physics_weight scheduler (ResidualCorrectionFNN is not in
        # PHYSICS_WEIGHT_MODELS — it has no physics loss, only an architecture
        # prior that α should stay near 1).
        la    = float(hp.get("alpha_reg_weight", 0.05))
        alpha = getattr(_loss_model, "_last_alpha", None)
        if la > 0 and alpha is not None:
            data_loss = data_loss + la * ((alpha - 1.0) ** 2).mean()
        return data_loss

    if model_type in {"PhysicsRegularizedFNN"}:
        _nj = tau_hat.shape[-1]
        _phys_total = _reduce_physics_to_total(physics, _nj)
        phys_reg = _loss_model.compute_loss(tau_hat, target, _phys_total)["physics"]
        if _norm is not None:
            _norm.update(float(data_loss.detach().item()), float(phys_reg.detach().item()))
            kappa = _norm.scale()
        else:
            kappa = 1.0
        return w_data * data_loss + w_phys * kappa * phys_reg

    raise ValueError(f"Unknown model_type for loss: {model_type!r}")


# =============================================================================
# Epoch-level training and evaluation
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    model_type: str,
    loss_w_d: float,
    loss_w_p: float,
    hp: dict,
    onecycle_sched=None,
    scaler=None,
    feature_noise_std: float = 0.0,
) -> tuple[float, float, float]:
    """Run one training epoch.

    Returns ``(mean_loss, mean_grad_norm, train_rmse_unweighted)``.

    Uses Automatic Mixed Precision (AMP) when ``scaler`` is not None.

    The joint weight tensor upweights J2 (shoulder) by 2.5× because it
    carries the full arm weight and shows the highest systematic residuals.
    """
    model.train()
    total_loss  = 0.0
    total_gnorm = 0.0
    total_sse   = 0.0
    total_elem  = 0
    use_amp     = scaler is not None
    n_batches   = len(loader)

    # Joint importance weighting: J2 (index 1) gets 2.5× weight in training loss
    _jw = torch.tensor([1.0, 2.5, 1.0, 1.0, 1.0], device=device)
    _grad_clip = float(hp.get("grad_clip_norm", 5.0))

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (features, target, physics) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        target   = target.to(device,   non_blocking=True)
        physics  = physics.to(device,  non_blocking=True)

        # Optional data-augmentation noise (training only)
        if feature_noise_std > 0.0:
            features = features + torch.randn_like(features) * feature_noise_std

        with torch.autocast(device_type=device.type, enabled=use_amp):
            tau_hat, components = _forward_pass(model, features, physics, model_type)
            loss = _compute_loss(
                tau_hat, target, physics, components,
                model, model_type, loss_w_d, loss_w_p, hp,
                _jw=_jw, features=features,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient clipping → optimiser step
        if scaler is not None:
            scaler.unscale_(optimizer)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_gnorm += gnorm.item() if hasattr(gnorm, "item") else float(gnorm)

        if onecycle_sched is not None:
            onecycle_sched.step()

        total_loss += loss.item()
        with torch.no_grad():
            d = tau_hat.detach() - target
            total_sse  += float((d * d).sum().item())
            total_elem += d.numel()

    train_rmse = math.sqrt(total_sse / max(1, total_elem))
    return total_loss / n_batches, total_gnorm / n_batches, train_rmse


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    model_type: str,
    hp: dict | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model on a data loader (no gradient computation).

    Always uses w_d=1 and w_p=0 (pure-data loss) so that validation loss is
    comparable across epochs regardless of the physics weight schedule.
    Early stopping tracks val_rmse (unweighted) by default.

    Returns ``(mean_loss, all_pred, all_target)`` in normalised space.
    """
    model.eval()
    total_loss = 0.0
    all_pred   = []
    all_target = []
    _hp = hp or {}

    for features, target, physics in loader:
        features = features.to(device, non_blocking=True)
        target   = target.to(device,   non_blocking=True)
        physics  = physics.to(device,  non_blocking=True)

        tau_hat, components = _forward_pass(model, features, physics, model_type)
        # No joint weighting in validation — unweighted MSE for fair cross-epoch comparison
        loss = _compute_loss(
            tau_hat, target, physics, components,
            model, model_type, 1.0, 0.0, _hp,
            _jw=None, features=features,
        )
        total_loss += loss.item()

        p    = tau_hat.cpu().numpy()
        t_np = target.cpu().numpy()
        if p.ndim == 3:
            # Sequence mode: flatten (B, T, J) → (B*T, J)
            p    = p.reshape(-1, p.shape[-1])
            t_np = t_np.reshape(-1, t_np.shape[-1])
        all_pred.append(p)
        all_target.append(t_np)

    return (
        total_loss / len(loader),
        np.concatenate(all_pred,   axis=0),
        np.concatenate(all_target, axis=0),
    )


# =============================================================================
# Artefact saving helpers
# =============================================================================

def save_comparison_plot(
    pred: np.ndarray,
    target: np.ndarray,
    metrics: dict,
    save_path: str,
    model_name: str,
) -> None:
    """Save a 5-panel per-joint prediction vs ground-truth matplotlib figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    joint_names = ["J1 (yaw)", "J2 (shoulder)", "J3 (elbow)",
                   "J4 (wrist)", "J5 (wrist roll)"]

    n_show = min(3000, len(pred))
    idx    = np.linspace(0, len(pred) - 1, n_show, dtype=int)

    for j in range(5):
        ax = axes[j]
        ax.plot(target[idx, j], color="darkorange", linewidth=1.0, label="Ground Truth")
        ax.plot(pred[idx, j],   color="steelblue",  linewidth=1.0, alpha=0.8,
                label=model_name)
        r2_j = metrics.get("r2",  [0] * 5)[j]
        ax.set_title(f"{joint_names[j]}  RMSE={metrics['rmse'][j]:.4f}  "
                     f"R²={r2_j:.4f}  NRMSE={metrics['nrmse'][j]:.3f}")
        ax.set_xlabel("sample")
        ax.set_ylabel("torque (N·m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Summary panel (6th subplot)
    ax = axes[5]
    ax.axis("off")
    _rp  = metrics.get("rmse_pooled", metrics["rmse_mean"])
    _r2o = metrics.get("r2_overall", metrics.get("r2_mean", 0.0))
    lines = [
        f"Model: {model_name}", "",
        f"Pooled RMSE:  {_rp:.5f} N·m  (all τ, same as val_rmse)",
        f"Macro RMSE:   {metrics['rmse_mean']:.5f} N·m  (mean of joint RMSE)",
        f"Mean MAE:     {metrics.get('mae_mean', 0):.5f} N·m",
        f"R2 overall:   {_r2o:.5f}   R2 macro: {metrics.get('r2_mean', 0):.5f}",
        f"Mean Pearson: {metrics.get('pearson_r_mean', 0):.5f}",
        f"Mean NRMSE:   {metrics['nrmse_mean']:.4f}", "",
        "Per-joint   RMSE      R2      MAE",
    ]
    for j in range(5):
        r2_j  = metrics.get("r2",  [0] * 5)[j]
        mae_j = metrics.get("mae", [0] * 5)[j]
        lines.append(f"  J{j+1}:  {metrics['rmse'][j]:.5f}  {r2_j:.4f}  {mae_j:.5f}")
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace")

    plt.suptitle(f"Prediction vs Ground Truth — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_architecture_summary(model: nn.Module, save_path: str) -> None:
    """Save a text file with model class, parameter count, and PyTorch repr."""
    lines = [
        f"Model: {model.__class__.__name__}",
        f"Params: {model.count_parameters():,}" if hasattr(model, "count_parameters") else "",
        "", "Hyperparameters:",
    ]
    _cfg = getattr(model, "hparams", None) or getattr(model, "config", None)
    if _cfg:
        for k, v in _cfg.items():
            lines.append(f"  {k}: {v}")
    lines += ["", "PyTorch repr:", str(model)]
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Registry update
# =============================================================================

def update_registry(
    registry_file: str,
    model_key: str,
    run_id: str,
    run_dir: str,
    hp: dict,
    metrics: dict,
    model_path: str,
    training_time_s: float = 0.0,
    device_str: str = "cpu",
    stopped_early: bool = False,
    epochs_ran: int = 0,
    epochs_max: int = 0,
    final_train_loss: float = 0.0,
    final_val_loss: float = 0.0,
    num_train_samples: int = 0,
    num_val_samples: int = 0,
    val_metrics: dict | None = None,
    test_metrics: dict | None = None,
    num_test_samples: int = 0,
) -> None:
    """Prepend a training run entry to the global models_registry.yaml.

    If the registry file is corrupted it is backed up before starting fresh.
    """
    try:
        registry: dict = {}
        if os.path.exists(registry_file):
            try:
                with open(registry_file, "r") as f:
                    registry = yaml.safe_load(f) or {}
            except Exception as load_err:
                backup = registry_file + ".bak"
                logger.warning("Registry unreadable (%s) — backing up to %s and starting fresh.",
                               load_err, backup)
                try:
                    shutil.copy2(registry_file, backup)
                except Exception:
                    pass
                registry = {}

        if "models" not in registry:
            registry["models"] = []

        entry = {
            "model_type": model_key,
            "run_id":     run_id,
            "run_dir":    run_dir,
            "trained_at": datetime.now().isoformat(),
            "model_path": model_path,
            "training": {
                "time_seconds":     round(training_time_s, 1),
                "time_formatted":   _fmt_time(training_time_s),
                "epochs_ran":       epochs_ran,
                "epochs_max":       epochs_max,
                "stopped_early":    stopped_early,
                "final_train_loss": round(final_train_loss, 7) if final_train_loss else None,
                "final_val_loss":   round(final_val_loss,   7) if final_val_loss   else None,
            },
            "hardware": {
                "device":        device_str,
                "torch_version": str(torch.__version__),
            },
            "data": {
                "num_train_samples": num_train_samples,
                "num_val_samples":   num_val_samples,
                "num_test_samples":  num_test_samples,
            },
            "hyperparams": _make_serializable(hp),
            # Headline "metrics" block: averaged val+test (inference on ALL
            # unseen data, pooled predictions).  This is what ranking uses.
            "metrics": {
                "avg_rmse_mean":    metrics.get("rmse_mean"),
                "avg_rmse_pooled":  metrics.get("rmse_pooled"),
                "avg_r2_overall":   metrics.get("r2_overall"),
                "avg_r2_mean":      metrics.get("r2_mean"),
                "avg_nrmse_mean":   metrics.get("nrmse_mean"),
                "avg_mse_mean":     metrics.get("mse_mean"),
                "avg_pearson_mean": metrics.get("pearson_r_mean"),
                "per_joint_rmse":   metrics.get("rmse"),
                "per_joint_r2":     metrics.get("r2"),
                # Back-compat aliases so historical tooling reading
                # "test_rmse_pooled" keeps working.
                "test_rmse_mean":   (test_metrics or metrics).get("rmse_mean"),
                "test_rmse_pooled": (test_metrics or metrics).get("rmse_pooled"),
                "test_r2_overall":  (test_metrics or metrics).get("r2_overall"),
                "test_nrmse_mean":  (test_metrics or metrics).get("nrmse_mean"),
                "test_mse_mean":    (test_metrics or metrics).get("mse_mean"),
            },
        }
        if val_metrics is not None:
            entry["val_metrics"] = {
                "rmse_pooled": val_metrics.get("rmse_pooled"),
                "rmse_mean":   val_metrics.get("rmse_mean"),
                "r2_overall":  val_metrics.get("r2_overall"),
                "r2_mean":     val_metrics.get("r2_mean"),
                "nrmse_mean":  val_metrics.get("nrmse_mean"),
                "per_joint_rmse": val_metrics.get("rmse"),
            }
        if test_metrics is not None:
            entry["test_metrics"] = {
                "rmse_pooled": test_metrics.get("rmse_pooled"),
                "rmse_mean":   test_metrics.get("rmse_mean"),
                "r2_overall":  test_metrics.get("r2_overall"),
                "r2_mean":     test_metrics.get("r2_mean"),
                "nrmse_mean":  test_metrics.get("nrmse_mean"),
                "per_joint_rmse": test_metrics.get("rmse"),
            }
        registry["models"].insert(0, entry)

        registry_out = {
            "total_models": len(registry["models"]),
            "last_updated": datetime.now().isoformat(),
            "models":       registry["models"],
        }
        os.makedirs(os.path.dirname(registry_file), exist_ok=True)
        with open(registry_file, "w") as f:
            yaml.dump(_make_serializable(registry_out), f,
                      default_flow_style=False, sort_keys=False)

    except Exception as e:
        logger.warning("Could not update registry: %s", e)


# =============================================================================
# Full training pipeline for one model
# =============================================================================

def train_model(
    run_dir: str,
    model_type: str,
    hp: dict,
    *,
    models_dir: str,
    registry_file: str,
    nn_dir: str,
    console=None,
    gpu_memory_fraction: float = 0.95,
    cuda_device: int | None = None,
    grid_progress: Any = None,
    grid_progress_key: str | None = None,
) -> tuple[str, dict]:
    """Full training pipeline for one model: data loading → training loop →
    evaluation → checkpoint saving → registry update.

    Parameters
    ----------
    run_dir      : path to the processed dataset directory (train/val/test CSVs)
    model_type   : model registry key (e.g. 'BlackBoxFNN')
    hp           : hyperparameter dict from gather_hp / get_default_hp
    models_dir   : base directory for saved model checkpoints
    registry_file: path to models_registry.yaml
    nn_dir       : Neural_Networks package directory (for compile cache)
    console      : optional Rich Console for progress output; if None plain
                   logger messages are used instead.
    gpu_memory_fraction : fraction of GPU memory this process may allocate
                          (0.0–1.0).  Lowered automatically when running
                          parallel grid-search workers on the same GPU.
    cuda_device         : if CUDA is available, logical device index for this
                          process (``torch.cuda.set_device``).  ``None`` → 0.
    grid_progress       : optional ``multiprocessing.Manager().dict()`` proxy
                          for parallel grid-search status (parent reads it).
    grid_progress_key   : key under which this run writes a small status dict.

    Returns
    -------
    (save_dir, test_metrics)
    """
    import psutil

    _log = console.print if console is not None else logger.info

    _gp = grid_progress
    _gk = grid_progress_key

    def _grid_prog(**kw: Any) -> None:
        if _gp is None or not _gk:
            return
        try:
            cur = dict(_gp[_gk]) if _gk in _gp else {}
            cur.update(kw)
            cur.setdefault("model", model_type)
            _gp[_gk] = cur
        except Exception:
            pass

    if torch.cuda.is_available():
        _dev = 0 if cuda_device is None else int(cuda_device)
        _nd = torch.cuda.device_count()
        _dev = max(0, min(_dev, _nd - 1))
        torch.cuda.set_device(_dev)
        device = torch.device("cuda", _dev)
    else:
        device = torch.device("cpu")
        _dev = -1

    # Deterministic seeding: in the reduced-data regime run-to-run init noise
    # dominates inter-model differences, so a fixed seed is required for
    # comparison tables to be meaningful.  Override via hp["seed"].
    _seed = int(hp.get("seed", 42))
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)

    _epochs_max = int(hp.get("epochs", 500))
    _grid_prog(
        stage="starting",
        epoch=0,
        epochs_max=_epochs_max,
        cell_id=str(hp.get("_grid_cell_id", "")),
        seed=int(hp.get("_grid_seed", 0) or 0),
        gpu=int(_dev),
    )

    if device.type == "cuda":
        # CUDA performance flags
        torch.backends.cudnn.benchmark        = True
        torch.backends.cudnn.deterministic    = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
        torch.cuda.empty_cache()
        # Silence noisy inductor / dynamo logs during compilation
        for _lg_name in ("torch._inductor", "torch._dynamo", "torch.fx",
                         "torch._inductor.select_algorithm"):
            logging.getLogger(_lg_name).setLevel(logging.ERROR)
        _cache_dir = os.path.join(nn_dir, ".torch_compile_cache")
        os.makedirs(_cache_dir, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _cache_dir)

    # ----- DataLoaders -------------------------------------------------------
    pin_memory  = device.type == "cuda"
    _ncpu       = os.cpu_count() or 4
    _ram_gb     = psutil.virtual_memory().total / 1e9
    _max_workers = 4 if _ram_gb < 20 else 8
    _nw_env = os.environ.get("NN_NUM_WORKERS", "").strip()
    num_workers = (max(0, int(_nw_env)) if _nw_env.isdigit()
                   else (max(2, min(_max_workers, _ncpu // 2))
                         if device.type == "cuda" else 0))
    _pf_env = os.environ.get("NN_PREFETCH", "").strip()
    _prefetch = (max(2, int(_pf_env)) if _pf_env.isdigit()
                 else (6 if _ram_gb < 20 else 10))

    from Neural_Networks.apps.loader import make_dataloaders
    loaders = make_dataloaders(
        run_dir         = run_dir,
        batch_size      = hp.get("batch_size", 2048),
        mode            = "pointwise",
        seq_len         = hp.get("seq_len", 50),
        stride          = hp.get("stride", 1),
        normalise       = True,
        num_workers     = num_workers,
        pin_memory      = pin_memory,
        prefetch_factor = _prefetch,
        drop_last       = True,
        data_train_fraction = float(hp.get("data_train_fraction", 1.0)),
        data_train_seed     = int(hp.get("data_train_seed",
                                          hp.get("_grid_seed", 0)) or 0),
    )
    _grid_prog(stage="data_ready", gpu=int(_dev))

    # ----- Model construction ------------------------------------------------
    model           = build_model(model_type, hp, device)
    _model_cls_name = model.__class__.__name__
    _model_n_params = model.count_parameters() if hasattr(model, "count_parameters") else 0

    # ----- torch.compile (optional) ------------------------------------------
    _compiled    = False
    _gpu_vram_gb = (torch.cuda.get_device_properties(
                        torch.cuda.current_device()).total_memory / 1e9
                    if device.type == "cuda" else 0)
    _compile_env = os.environ.get("NN_TORCH_COMPILE", "").strip().lower() in ("1", "true", "yes")
    _want_compile = (device.type == "cuda"
                     and (_gpu_vram_gb >= 16
                          or bool(hp.get("torch_compile", False))
                          or _compile_env))
    _compile_mode = str(hp.get("torch_compile_mode", "default")).strip().lower()
    if _compile_mode not in ("default", "reduce-overhead", "max-autotune"):
        _compile_mode = "default"
    if _want_compile:
        try:
            model     = torch.compile(model, mode=_compile_mode)
            _compiled = True
        except Exception:
            pass

    # ----- Optimiser & scheduler --------------------------------------------
    optimizer = build_optimizer(model, hp)

    n_train_batches = len(loaders["train"])

    onecycle_sched = None
    if hp.get("lr_scheduler", "reduce_on_plateau") == "onecycle":
        onecycle_sched = build_scheduler(optimizer, hp, n_train_batches)
        scheduler      = None
    else:
        scheduler = build_scheduler(optimizer, hp, n_train_batches)

    _phys_sched: PhysicsWeightScheduler | None = None
    if model_type in PHYSICS_WEIGHT_MODELS:
        _phys_sched = PhysicsWeightScheduler.from_hp(hp)
    scaler      = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    gpu_total_gb = (torch.cuda.get_device_properties(
                        torch.cuda.current_device()).total_memory / 1e9
                    if device.type == "cuda" else 0.0)
    _grid_prog(stage="training_ready", gpu=int(_dev))

    # ----- Early stopping state ---------------------------------------------
    _min_delta         = float(hp.get("min_delta", 1e-4))
    best_val_loss      = math.inf
    best_val_rmse      = math.inf
    best_val_rmse_phys = math.inf
    best_state: dict | None = None
    best_epoch_num     = 0
    patience_counter   = 0
    patience           = hp.get("patience", 100)
    use_early_stop     = hp.get("early_stopping", True)
    epochs             = int(hp.get("epochs", 500))
    _pe_raw = os.environ.get("NN_GRID_PROGRESS_EVERY", "1").strip()
    _prog_every = max(1, int(_pe_raw or "1"))
    _early_metric = str(hp.get("early_stop_metric", "val_rmse")).strip().lower()
    if _early_metric not in ("val_rmse", "val_loss"):
        _early_metric = "val_rmse"
    best_val_loss_track = math.inf

    _snapshot_every   = int(hp.get("snapshot_every", 0))
    _snapshots_saved: list[dict] = []

    # Cache directory for best-checkpoint saves during training
    _best_ckpt_cache_dir = os.path.join(
        models_dir, "._train_cache",
        f"{model_type}_{uuid.uuid4().hex[:10]}",
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_rmse": [], "val_rmse": [],
        "w_d": [], "w_p": [],
    }

    # Normalisation stats for converting val RMSE to physical units (N·m)
    _tau_std_val  = loaders["val"].dataset.std_tau
    _tau_mean_val = loaders["val"].dataset.mean_tau
    # Trajectory boundaries for macro-average RMSE across the validation set
    _val_trajectories: list[dict] = (
        loaders["val"].dataset.metadata
        .get("split", {}).get("stats", {}).get("val", {}).get("trajectories", [])
    )

    _grad_norm   = 0.0
    _wd_eff      = 1.0
    _wp_eff      = 0.0 if _phys_sched is None else _phys_sched.w_p
    stopped_early = False
    t0 = time.time()

    _log(f"  Training {model_type} for up to {epochs} epochs …")

    _grid_ep1_wall_reported = False
    _train_ep1_t0: float | None = None

    for epoch in range(1, epochs + 1):
        if epoch == 1:
            _train_ep1_t0 = time.time()
            _log(
                f"  epoch 1/{epochs}: first train+val pass "
                f"(often the slowest; compile / cache warmup) …"
            )
        # ── Training step ────────────────────────────────────────────────────
        train_loss, _grad_norm, train_rmse_unw = train_epoch(
            model, loaders["train"], optimizer,
            device, model_type,
            _wd_eff, _wp_eff, hp, onecycle_sched, scaler,
            feature_noise_std  = hp.get("feature_noise_std", 0.01),
        )

        # ── Validation step ───────────────────────────────────────────────────
        val_loss, _val_pred, _val_tgt = eval_epoch(
            model, loaders["val"], device, model_type, hp=hp,
        )
        # Canonical val_rmse: per-trajectory → per-joint RMSE → mean joints → mean trajectories.
        # Normalised space used for early-stopping comparison.
        _val_rmse_unw  = macro_rmse_numpy(_val_pred, _val_tgt, _val_trajectories)
        # Physical-space val_rmse (N·m) for display and physics-weight scheduler.
        _val_pred_phys = _val_pred * _tau_std_val + _tau_mean_val
        _val_tgt_phys  = _val_tgt  * _tau_std_val + _tau_mean_val
        _val_rmse_phys = macro_rmse_numpy(_val_pred_phys, _val_tgt_phys, _val_trajectories)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_rmse_unw)
        history["val_rmse"].append(_val_rmse_phys)   # store physical N·m for interpretable CSV
        history["w_d"].append(_wd_eff)
        history["w_p"].append(_wp_eff)

        if _phys_sched is not None:
            _wd_eff, _wp_eff = _phys_sched.step(epoch, _val_rmse_unw)

        # ── LR scheduler step ────────────────────────────────────────────────
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss if _early_metric == "val_loss" else _val_rmse_unw)
            elif onecycle_sched is None:
                scheduler.step()

        # ── Checkpoint selection ──────────────────────────────────────────────
        if _early_metric == "val_loss":
            improved = val_loss < (best_val_loss_track - 1e-7)
        else:
            improved = _val_rmse_unw < (best_val_rmse - _min_delta)

        if improved:
            if _early_metric == "val_loss":
                best_val_loss_track = val_loss
            best_val_loss      = val_loss
            best_val_rmse      = _val_rmse_unw
            best_val_rmse_phys = _val_rmse_phys
            best_state         = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
            best_epoch_num     = int(epoch)
            patience_counter   = 0
            if _early_metric == "val_rmse" and best_state is not None:
                try:
                    os.makedirs(_best_ckpt_cache_dir, exist_ok=True)
                    torch.save({
                        "state_dict": best_state, "epoch": int(epoch),
                        "val_rmse": float(_val_rmse_unw),
                        "model_type": model_type, "model_class": _model_cls_name,
                    }, os.path.join(_best_ckpt_cache_dir, "best_val_rmse.pt"))
                except OSError as _cache_e:
                    logger.warning("Could not write best val_rmse cache: %s", _cache_e)
        else:
            patience_counter += 1

        # ── Periodic lightweight snapshots (long runs / grid search) ───────
        if (_snapshot_every > 0 and epoch % _snapshot_every == 0
                and best_state is not None):
            _snap_dir = os.path.join(_best_ckpt_cache_dir, "snapshots")
            try:
                os.makedirs(_snap_dir, exist_ok=True)
                _rm_tag = float(best_val_rmse_phys)
                if not math.isfinite(_rm_tag):
                    _rm_tag = 0.0
                _snap_name = f"snapshot_ep{epoch:05d}_rmse{_rm_tag:.4f}.pt"
                torch.save({
                    "state_dict":     best_state,
                    "epoch":          int(best_epoch_num),
                    "snapshot_epoch": int(epoch),
                    "val_rmse":       float(best_val_rmse),
                    "val_rmse_phys":  float(best_val_rmse_phys),
                    "model_type":     model_type,
                    "model_class":    _model_cls_name,
                }, os.path.join(_snap_dir, _snap_name))
                _snapshots_saved.append({
                    "epoch":         int(epoch),
                    "best_epoch":    int(best_epoch_num),
                    "val_rmse_phys": float(best_val_rmse_phys),
                })
            except OSError as _snap_e:
                logger.warning("Could not write snapshot at epoch %d: %s",
                               epoch, _snap_e)

        if (epoch % _prog_every == 0 or epoch == 1 or epoch == epochs):
            _grid_prog(
                stage="training",
                epoch=int(epoch),
                epochs_max=int(epochs),
                val_rmse_phys=float(_val_rmse_phys),
                gpu=int(_dev),
                trial_elapsed_s=float(time.time() - t0),
            )
            _lr_disp = float("nan")
            try:
                _lr_disp = float(optimizer.param_groups[0]["lr"])
            except (IndexError, KeyError, TypeError, ValueError):
                pass
            _log(
                f"  epoch {epoch:>4}/{epochs}  "
                f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                f"val_rmse_phys={_val_rmse_phys:.5f} N·m  "
                f"w_p={_wp_eff:.4f}  lr={_lr_disp:.2e}  "
                f"best_ep={best_epoch_num}  patience={patience_counter}/{patience}"
            )

        if (
            epoch == 1
            and not _grid_ep1_wall_reported
            and _train_ep1_t0 is not None
            and _gp is not None
            and _gk
        ):
            _grid_prog(epoch1_wall_s=float(time.time() - _train_ep1_t0))
            _grid_ep1_wall_reported = True

        if use_early_stop and patience_counter >= patience:
            stopped_early = True
            break

    _grid_prog(
        stage="training_done",
        epoch=int(len(history["train_loss"])),
        epochs_max=int(epochs),
        stopped_early=bool(stopped_early),
        gpu=int(_dev),
        trial_elapsed_s=float(time.time() - t0),
    )

    # Capture final-epoch weights (before we overwrite with the best snapshot)
    final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    final_epoch_trained = len(history["train_loss"])

    _grid_prog(stage="final_eval", epoch=int(final_epoch_trained), gpu=int(_dev))

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0

    # ── Final evaluation ────────────────────────────────────────────────────
    # Val: evaluate on ENTIRE val split, unnormalise to physical units (N·m)
    # and keep arrays so we can pool with test for the averaged metric below.
    _, _val_pred_norm, _val_tgt_norm = eval_epoch(
        model, loaders["val"], device, model_type, hp=hp)
    _val_pred_phys = _val_pred_norm * _tau_std_val + _tau_mean_val
    _val_tgt_phys  = _val_tgt_norm  * _tau_std_val + _tau_mean_val
    val_metrics_final = compute_metrics(_val_pred_phys, _val_tgt_phys)

    # Test: evaluate on ENTIRE test split
    _, test_pred, test_target = eval_epoch(
        model, loaders["test"], device, model_type, hp=hp)
    _tau_std  = loaders["test"].dataset.std_tau
    _tau_mean = loaders["test"].dataset.mean_tau
    test_pred_phys   = test_pred   * _tau_std  + _tau_mean
    test_target_phys = test_target * _tau_std  + _tau_mean
    test_metrics     = compute_metrics(test_pred_phys, test_target_phys)

    # Averaged (val + test) metrics — concatenate the physical-unit
    # predictions and compute ONE metrics dict over all unseen data.  This
    # is the headline inference-quality metric used for the final ranking
    # because it reflects model behaviour on the full held-out distribution
    # rather than an arbitrary test-split slice.
    _eval_pred   = np.concatenate([_val_pred_phys, test_pred_phys],   axis=0)
    _eval_target = np.concatenate([_val_tgt_phys,  test_target_phys], axis=0)
    avg_metrics  = compute_metrics(_eval_pred, _eval_target)
    # Tag the eval source so downstream displays can label correctly.
    avg_metrics["_n_val"]  = int(len(_val_pred_phys))
    avg_metrics["_n_test"] = int(len(test_pred_phys))

    _log(
        f"  Inference RMSE (val+test pooled): "
        f"{avg_metrics['rmse_pooled']:.5f} N·m  "
        f"R²_ov = {avg_metrics['r2_overall']:.4f}   "
        f"[val n={avg_metrics['_n_val']}  test n={avg_metrics['_n_test']}]"
    )

    # ── Save artefacts ────────────────────────────────────────────────────────
    epochs_trained = len(history["train_loss"])
    rmse_val       = avg_metrics["rmse_pooled"]
    mse_val        = avg_metrics["mse_pooled"]
    from Neural_Networks.apps.checkpoint_io import build_run_id
    run_id = build_run_id(model_type, epochs_trained=epochs_trained,
                          rmse=rmse_val, hp=hp)
    save_dir = os.path.join(models_dir, MODEL_SAVE_DIRS[model_type], run_id)
    os.makedirs(save_dir, exist_ok=True)

    _snap_src = os.path.join(_best_ckpt_cache_dir, "snapshots")
    if os.path.isdir(_snap_src):
        _snap_dst = os.path.join(save_dir, "snapshots")
        try:
            shutil.move(_snap_src, _snap_dst)
        except OSError as _mv_e:
            logger.warning("Could not move snapshots directory: %s", _mv_e)

    # Normalisation stats — needed for inference without access to original dataset
    _train_ds = loaders["train"].dataset
    def _to_list(arr):
        return arr.tolist() if hasattr(arr, "tolist") else list(arr)
    _norm_stats = {
        "mean_tau": _to_list(_train_ds.mean_tau),
        "std_tau":  _to_list(_train_ds.std_tau),
        "mean_q":   _to_list(_train_ds.mean_q),
        "std_q":    _to_list(_train_ds.std_q),
        "mean_qd":  _to_list(_train_ds.mean_qd),
        "std_qd":   _to_list(_train_ds.std_qd),
        "mean_qdd": _to_list(_train_ds.mean_qdd),
        "std_qdd":  _to_list(_train_ds.std_qdd),
    }

    _unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
    _hparams_blob = (getattr(_unwrapped, "hparams", None)
                     or getattr(_unwrapped, "config", {}))

    from Neural_Networks.apps.checkpoint_io import save_checkpoints
    model_path, final_model_path = save_checkpoints(
        save_dir,
        model=model,
        final_state=final_state,
        best_epoch=best_epoch_num,
        epochs_trained=final_epoch_trained,
        model_cls_name=_model_cls_name,
        hparams_blob=_hparams_blob,
        norm_stats=_norm_stats,
        avg_metrics=avg_metrics,
        val_metrics=val_metrics_final,
        test_metrics=test_metrics,
    )

    from Neural_Networks.apps.checkpoint_io import dump_yaml, exhaustive_hparams
    meta_path = os.path.join(save_dir, "metadata.yaml")
    dump_yaml({
        "model_type":           model_type,
        "run_id":               run_id,
        "data_run_dir":         run_dir,
        "trained_at":           datetime.now().isoformat(),
        "device":               str(device),
        "epochs_trained":       int(epochs_trained),
        "best_val_loss":        float(best_val_loss),
        "best_val_rmse":        float(best_val_rmse),
        "hyperparams":          dict(hp),
        "exhaustive_hyperparams": exhaustive_hparams(
            model_type, hp,
            n_train_samples=int(hp.get("_n_train_samples", 0) or 0),
        ),
        "physics_sched_config": (_phys_sched.config_dict()
                                 if _phys_sched is not None else None),
        # "metrics" is now the averaged (val+test) block — the headline
        # inference-quality number.  Per-split details live in the
        # dedicated "val_metrics" and "test_metrics" keys.
        "metrics":              avg_metrics,
        "val_metrics":          val_metrics_final,
        "test_metrics":         test_metrics,
        "eval_sources": {
            "val_samples":  int(avg_metrics.get("_n_val",  0)),
            "test_samples": int(avg_metrics.get("_n_test", 0)),
            "note": "metrics block is compute_metrics over concatenated val+test predictions in N·m",
        },
    }, meta_path)

    save_comparison_plot(
        _eval_pred,
        _eval_target,
        avg_metrics,
        os.path.join(save_dir, "comparison_plot.png"),
        model_type,
    )
    save_architecture_summary(_unwrapped, os.path.join(save_dir, "architecture.txt"))

    # Training history plot + CSV
    _hist_png = os.path.join(save_dir, "training_history.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="train loss", color="steelblue")
    ax.plot(history["val_loss"],   label="val loss",   color="darkorange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"Training History — {model_type}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_hist_png, dpi=100)
    plt.close(fig)

    _csv_path = os.path.join(save_dir, "training_history.csv")
    _hist_wd  = history.get("w_d") or []
    _hist_wp  = history.get("w_p") or []
    with open(_csv_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "train_rmse", "val_rmse", "w_d", "w_p"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.8f}",
                f"{history['val_loss'][i]:.8f}",
                f"{history['train_rmse'][i]:.8f}",
                f"{history['val_rmse'][i]:.8f}",
                f"{float(_hist_wd[i]) if i < len(_hist_wd) else 1.0:.8f}",
                f"{float(_hist_wp[i]) if i < len(_hist_wp) else 0.0:.8f}",
            ])

    device_str = (f"cuda:{torch.cuda.get_device_name(0)}"
                  if device.type == "cuda" else "cpu")
    update_registry(
        registry_file      = registry_file,
        model_key          = model_type,
        run_id             = run_id,
        run_dir            = save_dir,
        hp                 = hp,
        metrics            = avg_metrics,
        val_metrics        = val_metrics_final,
        test_metrics       = test_metrics,
        model_path         = model_path,
        training_time_s    = elapsed,
        device_str         = device_str,
        stopped_early      = stopped_early,
        epochs_ran         = epochs_trained,
        epochs_max         = epochs,
        final_train_loss   = history["train_loss"][-1] if history["train_loss"] else 0.0,
        final_val_loss     = history["val_loss"][-1]   if history["val_loss"]   else 0.0,
        num_train_samples  = len(loaders["train"].dataset),
        num_val_samples    = len(loaders["val"].dataset),
        num_test_samples   = len(loaders["test"].dataset),
    )

    _log(f"  Saved to: {save_dir}")
    # Return the averaged metrics so the ranking table sorts by val+test
    # pooled RMSE.  Per-split dicts are stashed on the returned mapping for
    # display code that wants to show them side-by-side.
    avg_metrics["_val_metrics"]  = val_metrics_final
    avg_metrics["_test_metrics"] = test_metrics
    # Training metadata for display in ranking tables
    avg_metrics["_epochs_trained"]     = int(epochs_trained)
    avg_metrics["_epochs_max"]         = int(epochs)
    avg_metrics["_best_epoch"]         = int(best_epoch_num)
    avg_metrics["_stopped_early"]      = stopped_early
    avg_metrics["_best_val_rmse_phys"] = float(best_val_rmse_phys)
    avg_metrics["_lr_scheduler"]       = str(hp.get("lr_scheduler", ""))
    avg_metrics["_physics_weight"]     = float(hp.get("physics_weight", 0.0))
    avg_metrics["_weight_decay"]       = float(hp.get("weight_decay", 0.0))
    avg_metrics["_snapshots"]        = list(_snapshots_saved)
    avg_metrics["_snapshot_epochs"]  = [int(s["epoch"]) for s in _snapshots_saved]
    return save_dir, avg_metrics
