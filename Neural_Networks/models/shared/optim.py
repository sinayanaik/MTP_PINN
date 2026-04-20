"""AdamW factory and LR schedulers shared by torque trainers."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


def build_optimizer_default(model: torch.nn.Module, hp: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(hp.get("learning_rate", 3e-4))
    wd = float(hp.get("weight_decay", 2e-3))
    return AdamW(model.parameters(), lr=lr, weight_decay=wd)


def build_optimizer_physics_regularized(model: Any, hp: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(hp.get("learning_rate", 3e-4))
    wd = float(hp.get("weight_decay", 2e-3))
    phi_lr = lr * float(hp.get("phi_lr_ratio", 0.1))
    return AdamW(
        [
            {"params": list(model.net.parameters()), "lr": lr, "weight_decay": wd},
            {"params": [model.tau_scale, model.tau_bias], "lr": phi_lr, "weight_decay": 0.0},
        ]
    )


def build_scheduler(optimizer, hp: dict[str, Any], n_train_batches: int):
    sched_name = str(hp.get("lr_scheduler", "reduce_on_plateau")).lower()
    epochs = int(hp.get("epochs", 1000))
    lr = float(hp.get("learning_rate", 3e-4))
    if sched_name == "none":
        return None
    if sched_name == "warmup_cosine":
        warmup_ep = max(1, epochs // 20)
        min_factor = 0.01

        def _warmup_cosine_lambda(ep: int) -> float:
            if ep < warmup_ep:
                return 0.1 + 0.9 * (ep / warmup_ep)
            progress = (ep - warmup_ep) / max(1, epochs - warmup_ep)
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, _warmup_cosine_lambda)
    if sched_name == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, patience=25, factor=0.5, min_lr=lr * 0.01, threshold=1e-4)
    if sched_name == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR

        return OneCycleLR(optimizer, max_lr=lr * 10, total_steps=epochs * n_train_batches)
    return None
