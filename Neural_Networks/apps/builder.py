"""
Neural_Networks.core.builder
==============================
Factory functions for constructing models, optimisers, and LR schedulers
from hyperparameter dicts.

These functions are the single source of truth for the
``hp → PyTorch object`` mapping.  They carry no UI or logging side-effects
so that they can be called from training scripts, notebooks, or tests.

Public API
----------
build_model(model_type, hp, device)     -> nn.Module
build_optimizer(model, hp)              -> torch.optim.Optimizer
build_scheduler(optimizer, hp, n_train_batches) -> scheduler | None
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, RMSprop, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicLR,
    ExponentialLR, OneCycleLR, ReduceLROnPlateau, StepLR,
)

from Neural_Networks.models import MODEL_REGISTRY
from Neural_Networks.physics import ACTIVE_JOINTS


def build_model(model_type: str, hp: dict, device: torch.device) -> nn.Module:
    """Instantiate the correct model class with the given hyperparameter dict.

    Each model type maps to a specific constructor signature; this function
    translates the flat HP dict into the appropriate keyword arguments.

    Parameters
    ----------
    model_type : str
        One of the registered model type names in MODEL_REGISTRY.
    hp : dict
        Hyperparameter dict as returned by gather_hp / get_default_hp.
    device : torch.device
        Device to move the model to after construction.

    Returns
    -------
    nn.Module  (on *device*)
    """
    ModelClass = MODEL_REGISTRY[model_type]
    n_joints   = ACTIVE_JOINTS  # 5 active joints for the Kikobot arm

    if model_type == "PhysicsRegularizedFNN":
        kwargs: dict = {
            "n_joints":      n_joints,
            "hidden_layers": hp.get("hidden_layers", [256, 256, 128]),
            "dropout":       hp.get("dropout", 0.1),
            "activation":    hp.get("activation", "silu"),
        }

    elif model_type == "ResidualCorrectionFNN":
        kwargs = {
            "n_joints":      n_joints,
            "hidden_layers": hp.get("hidden_layers", [256, 256, 128]),
            "dropout":       hp.get("dropout", 0.1),
            "activation":    hp.get("activation", "tanh"),
        }

    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    return ModelClass(**kwargs).to(device)


def build_optimizer(model: nn.Module, hp: dict) -> torch.optim.Optimizer:
    """Build an optimiser from the HP dict.

    Handles split parameter groups when the model has a ``tau_calib``
    sub-module (e.g. PhysicsRegularizedFNN) — calibration uses a lower LR
    scaled by ``phi_lr_ratio``.

    Supported optimisers: ``adam``, ``adamw`` (default), ``sgd``, ``rmsprop``.
    """
    opt_name  = hp.get("optimizer", "adamw").lower()
    lr        = hp.get("learning_rate", 3e-4)
    wd        = hp.get("weight_decay",  2e-3)
    phi_ratio = float(hp.get("phi_lr_ratio", 0.1))

    # Split main vs. calibration params for EC-PINN
    calib_params: list[nn.Parameter] = []
    main_params:  list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("tau_calib."):
            calib_params.append(p)
        else:
            main_params.append(p)

    if calib_params:
        # Calibration layers: lower LR, no weight decay (they are scale+bias terms)
        param_groups = [
            {"params": main_params,  "lr": lr,             "weight_decay": wd},
            {"params": calib_params, "lr": lr * phi_ratio, "weight_decay": 0.0},
        ]
    else:
        param_groups = [{"params": list(model.parameters()), "lr": lr, "weight_decay": wd}]

    opts = {
        "adam":    lambda: Adam(param_groups),
        "adamw":   lambda: AdamW(param_groups),
        "sgd":     lambda: SGD(param_groups, momentum=0.9),
        "rmsprop": lambda: RMSprop(param_groups),
    }
    return opts.get(opt_name, opts["adamw"])()


def build_scheduler(optimizer, hp: dict, n_train_batches: int):
    """Build the LR scheduler (or return None for ``none`` mode).

    For ``onecycle``, the caller should pass ``onecycle_sched`` separately
    and call it per-batch; all other schedulers are epoch-level.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    hp : dict
    n_train_batches : int
        Number of batches per epoch (required for OneCycleLR and CyclicLR).
    """
    sched_name = hp.get("lr_scheduler", "reduce_on_plateau").lower()
    epochs     = hp.get("epochs", 1000)
    lr         = hp.get("learning_rate", 3e-4)

    if sched_name == "none":
        return None

    if sched_name == "warmup_cosine":
        warmup_ep  = max(1, epochs // 20)
        min_factor = 0.01

        def _warmup_cosine_lambda(ep: int) -> float:
            # Linear ramp for first 5% of epochs, then cosine decay to 1% of LR.
            # Starts at 10% of peak (not 5%) so first epochs are productive.
            if ep < warmup_ep:
                return 0.1 + 0.9 * (ep / warmup_ep)
            progress = (ep - warmup_ep) / max(1, epochs - warmup_ep)
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_cosine_lambda)

    if sched_name == "warmup_cosine_restarts":
        # Cosine annealing with warm restarts: periodic LR resets allow the
        # optimizer to escape local minima.  A linear warmup ramp precedes the
        # first cosine cycle so early gradients are stable.
        #
        # Schedule: linear ramp 0.1*lr → lr over warmup_ep epochs, then
        # CosineAnnealingWarmRestarts with T_0 and T_mult from HP.
        warmup_ep  = max(1, epochs // 20)           # 5% warmup
        t_0        = int(hp.get("restart_period", 40))   # first cycle length
        t_mult     = int(hp.get("restart_mult", 2))      # cycle length multiplier
        min_factor = 0.01

        base_sched = CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=t_mult, eta_min=lr * min_factor,
        )

        def _warmup_restart_lambda(ep: int) -> float:
            if ep < warmup_ep:
                return 0.1 + 0.9 * (ep / warmup_ep)
            return 1.0  # base_sched handles the cosine after warmup

        warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_restart_lambda)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, base_sched], milestones=[warmup_ep],
        )

    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    if sched_name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 5), gamma=0.5)

    if sched_name == "reduce_on_plateau":
        # patience=25: buffer so physics-term noise doesn't prematurely halve LR.
        # factor=0.5: moderate reduction. threshold=1e-4: react only to real changes.
        return ReduceLROnPlateau(optimizer, patience=25, factor=0.5,
                                 min_lr=lr * 0.01, threshold=1e-4)

    if sched_name == "onecycle":
        return OneCycleLR(optimizer, max_lr=lr * 10,
                          total_steps=epochs * n_train_batches)

    if sched_name == "exponential":
        return ExponentialLR(optimizer, gamma=0.95)

    if sched_name == "cyclic":
        return CyclicLR(optimizer, base_lr=lr * 0.1, max_lr=lr,
                        step_size_up=n_train_batches * 5, mode="triangular2")

    return None
