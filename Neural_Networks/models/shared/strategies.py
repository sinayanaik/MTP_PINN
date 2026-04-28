"""Per-architecture hooks: model construction, optimiser, train/eval steps, run-id keys."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Train-epoch return type
# ---------------------------------------------------------------------------
#
# Strategies return a ``TrainEpochMetrics`` so the pipeline can derive
# physically-meaningful train_rmse (N·m) and a train_loss that is directly
# comparable to val_loss.  The two key invariants exposed here:
#
#   loss_data_unw : unweighted per-element MSE on the **target** torque
#                   (same formulation as the per-strategy eval_epoch's
#                   val_loss → directly comparable on the same plot).
#   sse_per_joint : per-joint summed squared errors in **normalised**
#                   target space (shape (n_joints,)).  Multiplied by
#                   ``std_tau_per_joint`` in the pipeline to obtain the
#                   physical-units macro RMSE that matches val_rmse.
#
# ``loss_total`` remains the actual objective being optimised (joint-weighted
# MSE for the data-only baseline, blended (1-α)·data + α·phys for PhysReg,
# data + α_reg·||δ||² for Residual, …).  The pipeline records it as
# ``train_loss_obj`` for diagnostics but does NOT plot it against val_loss
# (their formulations differ).
@dataclass(frozen=True)
class TrainEpochMetrics:
    loss_total:    float
    loss_data_unw: float
    grad_norm:     float
    sse_per_joint: np.ndarray
    n_samples:     int
    extras:        dict[str, Any] | None = None


def _accumulate_sse(running: np.ndarray | None, batch_sse: np.ndarray) -> np.ndarray:
    return batch_sse if running is None else running + batch_sse

from Neural_Networks.loader import ACTIVE_JOINTS
from Neural_Networks.models.shared.optim import build_optimizer_default
from Neural_Networks.models.torque_models import (
    BlackBoxFNN,
    PhysicsRegularizedFNN,
    ResidualCorrectionFNN,
    reduce_physics_to_total,
)

# --- default exhaustive_hp bases (merged with run hp for metadata) ----------

DEFAULT_EXHAUSTIVE_PLAIN: dict[str, Any] = {
    "batch_size": 512,
    "epochs": 500,
    "learning_rate": 3e-4,
    "lr_scheduler": "warmup_cosine",
    "weight_decay": 5e-2,
    "dropout": 0.2,
    "activation": "silu",
    "hidden_layers": [128, 256, 128],
    "early_stopping": True,
    "early_stop_metric": "val_rmse",
    "patience": 60,
    "min_delta": 1e-4,
    "grad_clip_norm": 1.0,
    "feature_noise_std": 0.05,
    "data_train_fraction": 1.0,
    "data_train_seed": 0,
    "stride": 1,
    "seq_len": 50,
    "torch_compile": False,
    "torch_compile_mode": "default",
    "seed": 42,
}

DEFAULT_EXHAUSTIVE_PHYSICS_REG = {
    **DEFAULT_EXHAUSTIVE_PLAIN,
    "physics_weight": 0.1,
    "physics_warmup_fraction": 0.05,
}

DEFAULT_EXHAUSTIVE_RESIDUAL = {
    **DEFAULT_EXHAUSTIVE_PLAIN,
    "alpha_reg_weight": 0.05,
    "correction_scale": 0.5,
}

RUN_ID_KEYS_PLAIN: list[tuple[str, str]] = [
    ("data_train_fraction", "frac"),
    ("learning_rate", "lr"),
    ("weight_decay", "wd"),
    ("dropout", "do"),
    ("batch_size", "bs"),
    ("hidden_layers", "hl"),
]

RUN_ID_KEYS_PHYSICS_REG = [
    *RUN_ID_KEYS_PLAIN,
    ("physics_weight", "pw"),
]

RUN_ID_KEYS_RESIDUAL = [
    *RUN_ID_KEYS_PLAIN,
    ("alpha_reg_weight", "alphareg"),
]


def _loss_mse(tau_hat: torch.Tensor, target: torch.Tensor, joint_weights: torch.Tensor | None) -> torch.Tensor:
    if joint_weights is None:
        return F.mse_loss(tau_hat, target)
    return (joint_weights * (tau_hat - target) ** 2).mean()


def _make_model_plain(device: torch.device, hp: dict[str, Any]) -> nn.Module:
    return BlackBoxFNN(
        n_joints=ACTIVE_JOINTS,
        hidden_layers=list(hp.get("hidden_layers", [128, 256, 128])),
        dropout=float(hp.get("dropout", 0.2)),
        activation=str(hp.get("activation", "silu")),
    ).to(device)


def _train_epoch_plain(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    hp: dict[str, Any],
    _epoch: int,
    onecycle_sched,
    scaler,
) -> TrainEpochMetrics:
    model.train()
    total_loss = 0.0
    total_loss_data_unw = 0.0
    total_gnorm = 0.0
    sse_per_joint: np.ndarray | None = None
    n_samples = 0
    use_amp = scaler is not None
    n_batches = len(loader)
    _jw = torch.tensor([1.0, 2.5, 1.0, 1.0, 1.0], device=device)
    _grad_clip = float(hp.get("grad_clip_norm", 5.0))
    optimizer.zero_grad(set_to_none=True)
    for _batch_idx, (features, target, physics) in enumerate(loader):
        del physics
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if float(hp.get("feature_noise_std", 0.0) or 0.0) > 0.0:
            features = features + torch.randn_like(features) * float(hp["feature_noise_std"])
        with torch.autocast(device_type=device.type, enabled=use_amp):
            tau_hat = model(features, None)
            loss = _loss_mse(tau_hat, target, _jw)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_gnorm += gnorm.item() if hasattr(gnorm, "item") else float(gnorm)
        if onecycle_sched is not None:
            onecycle_sched.step()
        total_loss += float(loss.item())
        with torch.no_grad():
            d = tau_hat.detach() - target
            sse_per_joint = _accumulate_sse(sse_per_joint, (d * d).sum(dim=0).cpu().numpy())
            n_samples += int(d.shape[0])
            total_loss_data_unw += float(F.mse_loss(tau_hat.detach(), target).item())
        del tau_hat, loss, d
    return TrainEpochMetrics(
        loss_total=total_loss / n_batches,
        loss_data_unw=total_loss_data_unw / n_batches,
        grad_norm=total_gnorm / n_batches,
        sse_per_joint=sse_per_joint if sse_per_joint is not None else np.zeros(ACTIVE_JOINTS),
        n_samples=n_samples,
        extras=None,
    )


def _eval_epoch_plain(model: nn.Module, loader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_pred: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    with torch.no_grad():
        for features, target, physics in loader:
            del physics
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            tau_hat = model(features, None)
            loss = _loss_mse(tau_hat, target, None)
            total_loss += loss.item()
            p = tau_hat.cpu().numpy()
            t_np = target.cpu().numpy()
            if p.ndim == 3:
                p = p.reshape(-1, p.shape[-1])
                t_np = t_np.reshape(-1, t_np.shape[-1])
            all_pred.append(p)
            all_target.append(t_np)
            del features, target, tau_hat, loss, p, t_np
    return total_loss / len(loader), np.concatenate(all_pred, axis=0), np.concatenate(all_target, axis=0)


def _make_model_physics_reg(device: torch.device, hp: dict[str, Any]) -> nn.Module:
    return PhysicsRegularizedFNN(
        n_joints=ACTIVE_JOINTS,
        hidden_layers=list(hp.get("hidden_layers", [128, 256, 128])),
        dropout=float(hp.get("dropout", 0.2)),
        activation=str(hp.get("activation", "silu")),
    ).to(device)


def _train_epoch_physics_reg(
    model: PhysicsRegularizedFNN,
    loader,
    optimizer,
    device: torch.device,
    hp: dict[str, Any],
    epoch: int,
    onecycle_sched,
    scaler,
) -> TrainEpochMetrics:
    model.train()
    total_loss = 0.0
    total_l_data_jw = 0.0
    total_l_phys_jw = 0.0
    total_loss_data_unw = 0.0
    total_gnorm = 0.0
    sse_per_joint: np.ndarray | None = None
    n_samples = 0
    use_amp = scaler is not None
    n_batches = len(loader)
    _jw = torch.tensor([1.0, 2.5, 1.0, 1.0, 1.0], device=device)
    _grad_clip = float(hp.get("grad_clip_norm", 5.0))
    epochs_max = max(1, int(hp.get("epochs", 500)))
    warmup_ep = max(1, int(float(hp.get("physics_warmup_fraction", 0.05)) * epochs_max))
    pw = float(hp.get("physics_weight", 0.1))
    if pw < 0.0:
        raise ValueError(f"physics_weight must be >= 0.0 (got {pw}).")
    # Additive penalty coefficient — ramps from 0 → pw over warmup.
    # Loss = L_data + alpha_eff * L_phys (Tikhonov form: data always fully weighted).
    alpha_eff = pw * min(1.0, float(epoch) / float(warmup_ep))
    optimizer.zero_grad(set_to_none=True)
    for _batch_idx, (features, target, physics) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        physics = physics.to(device, non_blocking=True)
        if float(hp.get("feature_noise_std", 0.0) or 0.0) > 0.0:
            features = features + torch.randn_like(features) * float(hp["feature_noise_std"])
        with torch.autocast(device_type=device.type, enabled=use_amp):
            tau_hat = model(features, physics)
            tau_ref = reduce_physics_to_total(physics, model.n_joints)
            l_data = _loss_mse(tau_hat, target, _jw)
            l_phys = _loss_mse(tau_hat, tau_ref, _jw)
            # Additive Tikhonov penalty: data loss is always fully weighted;
            # physics acts as a regulariser pulling predictions toward RNEA.
            loss = l_data + alpha_eff * l_phys
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_gnorm += gnorm.item() if hasattr(gnorm, "item") else float(gnorm)
        if onecycle_sched is not None:
            onecycle_sched.step()
        total_loss += float(loss.item())
        total_l_data_jw += float(l_data.item())
        total_l_phys_jw += float(l_phys.item())
        with torch.no_grad():
            d = tau_hat.detach() - target
            sse_per_joint = _accumulate_sse(sse_per_joint, (d * d).sum(dim=0).cpu().numpy())
            n_samples += int(d.shape[0])
            total_loss_data_unw += float(F.mse_loss(tau_hat.detach(), target).item())
        del tau_hat, loss, d, l_data, l_phys, tau_ref
    return TrainEpochMetrics(
        loss_total=total_loss / n_batches,
        loss_data_unw=total_loss_data_unw / n_batches,
        grad_norm=total_gnorm / n_batches,
        sse_per_joint=sse_per_joint if sse_per_joint is not None else np.zeros(ACTIVE_JOINTS),
        n_samples=n_samples,
        extras={
            "l_data_jw": total_l_data_jw / n_batches,
            "l_phys_jw": total_l_phys_jw / n_batches,
            "alpha_eff": float(alpha_eff),
            "physics_weight": float(pw),
            "loss_form": "additive",
        },
    )


def _eval_epoch_physics_reg(
    model: PhysicsRegularizedFNN, loader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_pred: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    with torch.no_grad():
        for features, target, physics in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            physics = physics.to(device, non_blocking=True)
            tau_hat = model(features, physics)
            loss = _loss_mse(tau_hat, target, None)
            total_loss += loss.item()
            p = tau_hat.cpu().numpy()
            t_np = target.cpu().numpy()
            if p.ndim == 3:
                p = p.reshape(-1, p.shape[-1])
                t_np = t_np.reshape(-1, t_np.shape[-1])
            all_pred.append(p)
            all_target.append(t_np)
            del features, target, physics, tau_hat, loss, p, t_np
    return total_loss / len(loader), np.concatenate(all_pred, axis=0), np.concatenate(all_target, axis=0)


def _make_model_residual(device: torch.device, hp: dict[str, Any]) -> nn.Module:
    return ResidualCorrectionFNN(
        n_joints=ACTIVE_JOINTS,
        hidden_layers=list(hp.get("hidden_layers", [128, 256, 128])),
        dropout=float(hp.get("dropout", 0.2)),
        activation=str(hp.get("activation", "silu")),
        correction_scale=float(hp.get("correction_scale", 0.5)),
    ).to(device)


def _train_epoch_residual(
    model: ResidualCorrectionFNN,
    loader,
    optimizer,
    device: torch.device,
    hp: dict[str, Any],
    _epoch: int,
    onecycle_sched,
    scaler,
) -> TrainEpochMetrics:
    model.train()
    total_loss = 0.0
    total_loss_data_unw = 0.0
    total_gnorm = 0.0
    sse_per_joint: np.ndarray | None = None
    n_samples = 0
    use_amp = scaler is not None
    n_batches = len(loader)
    _jw = torch.tensor([1.0, 2.5, 1.0, 1.0, 1.0], device=device)
    _grad_clip = float(hp.get("grad_clip_norm", 5.0))
    ar = float(hp.get("alpha_reg_weight", 0.05))
    delta_abs_sum = 0.0
    tau_phys_abs_sum = 0.0
    delta_denom = 0
    optimizer.zero_grad(set_to_none=True)
    for _batch_idx, (features, target, physics) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        physics = physics.to(device, non_blocking=True)
        if float(hp.get("feature_noise_std", 0.0) or 0.0) > 0.0:
            features = features + torch.randn_like(features) * float(hp["feature_noise_std"])
        with torch.autocast(device_type=device.type, enabled=use_amp):
            tau_phys = reduce_physics_to_total(physics, model.n_joints)
            # model.forward handles feat_aug construction and tanh bounding.
            # Extract bounded delta from the output for L2 regularisation.
            tau_hat = model(features, physics)
            delta = tau_hat - tau_phys
            loss = _loss_mse(tau_hat, target, _jw) + ar * (delta**2).mean()
        with torch.no_grad():
            delta_abs_sum += float(delta.detach().abs().sum().item())
            tau_phys_abs_sum += float(tau_phys.detach().abs().sum().item())
            delta_denom += delta.numel()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_gnorm += gnorm.item() if hasattr(gnorm, "item") else float(gnorm)
        if onecycle_sched is not None:
            onecycle_sched.step()
        total_loss += float(loss.item())
        with torch.no_grad():
            d = tau_hat.detach() - target
            sse_per_joint = _accumulate_sse(sse_per_joint, (d * d).sum(dim=0).cpu().numpy())
            n_samples += int(d.shape[0])
            total_loss_data_unw += float(F.mse_loss(tau_hat.detach(), target).item())
        del tau_hat, loss, d, delta, tau_phys  # no feat_aug (now inside model.forward)
    delta_ratio: float | None = None
    if delta_denom > 0 and tau_phys_abs_sum > 0.0:
        delta_ratio = (delta_abs_sum / delta_denom) / ((tau_phys_abs_sum / delta_denom) + 1e-12)
        logger.info("residual δ-ratio  E[|δ|]/E[|τ_phys|] = %.4f  (alpha_reg=%.3f)", delta_ratio, ar)
    return TrainEpochMetrics(
        loss_total=total_loss / n_batches,
        loss_data_unw=total_loss_data_unw / n_batches,
        grad_norm=total_gnorm / n_batches,
        sse_per_joint=sse_per_joint if sse_per_joint is not None else np.zeros(ACTIVE_JOINTS),
        n_samples=n_samples,
        extras={"alpha_reg_weight": ar, "delta_ratio": delta_ratio},
    )


def _eval_epoch_residual(
    model: ResidualCorrectionFNN, loader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_pred: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    with torch.no_grad():
        for features, target, physics in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            physics = physics.to(device, non_blocking=True)
            tau_hat = model(features, physics)
            loss = _loss_mse(tau_hat, target, None)
            total_loss += loss.item()
            p = tau_hat.cpu().numpy()
            t_np = target.cpu().numpy()
            if p.ndim == 3:
                p = p.reshape(-1, p.shape[-1])
                t_np = t_np.reshape(-1, t_np.shape[-1])
            all_pred.append(p)
            all_target.append(t_np)
            del features, target, physics, tau_hat, loss, p, t_np
    return total_loss / len(loader), np.concatenate(all_pred, axis=0), np.concatenate(all_target, axis=0)


def _sched_plain(_hp: dict[str, Any]) -> None:
    return None


def _sched_physics_reg(hp: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "linear_warmup",
        "weight": float(hp.get("physics_weight", 0.1)),
        "warmup_epochs": max(
            1, int(float(hp.get("physics_warmup_fraction", 0.05)) * int(hp.get("epochs", 500)))
        ),
    }


@dataclass(frozen=True)
class TorqueTrainStrategy:
    default_exhaustive_hp: dict[str, Any]
    run_id_hp_keys: list[tuple[str, str]]
    make_model: Callable[[torch.device, dict[str, Any]], nn.Module]
    build_optimizer: Callable[[nn.Module, dict[str, Any]], torch.optim.Optimizer]
    train_epoch: Callable[..., TrainEpochMetrics]
    eval_epoch: Callable[[nn.Module, Any, torch.device], tuple[float, np.ndarray, np.ndarray]]
    physics_sched_metadata: Callable[[dict[str, Any]], dict[str, Any] | None]


PLAIN_STRATEGY = TorqueTrainStrategy(
    default_exhaustive_hp=DEFAULT_EXHAUSTIVE_PLAIN,
    run_id_hp_keys=RUN_ID_KEYS_PLAIN,
    make_model=_make_model_plain,
    build_optimizer=build_optimizer_default,
    train_epoch=_train_epoch_plain,
    eval_epoch=_eval_epoch_plain,
    physics_sched_metadata=_sched_plain,
)

PHYSICS_REG_STRATEGY = TorqueTrainStrategy(
    default_exhaustive_hp=DEFAULT_EXHAUSTIVE_PHYSICS_REG,
    run_id_hp_keys=RUN_ID_KEYS_PHYSICS_REG,
    make_model=_make_model_physics_reg,
    build_optimizer=build_optimizer_default,
    train_epoch=_train_epoch_physics_reg,
    eval_epoch=_eval_epoch_physics_reg,
    physics_sched_metadata=_sched_physics_reg,
)

RESIDUAL_STRATEGY = TorqueTrainStrategy(
    default_exhaustive_hp=DEFAULT_EXHAUSTIVE_RESIDUAL,
    run_id_hp_keys=RUN_ID_KEYS_RESIDUAL,
    make_model=_make_model_residual,
    build_optimizer=build_optimizer_default,
    train_epoch=_train_epoch_residual,
    eval_epoch=_eval_epoch_residual,
    physics_sched_metadata=_sched_plain,
)
