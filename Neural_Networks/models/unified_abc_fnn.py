"""
Unified feedforward models for Categories A (black-box), B (soft physics loss), C (residual on tau_phys).

Single implementation; registry exposes BlackBoxFNN / PhysicsRegularizedFNN / ResidualCorrectionFNN
as thin subclasses fixing physics_mode.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from Neural_Networks.models.common import ACTIVATION_MAP, build_mlp
from Neural_Networks.models.tau_equation_calibration import TauEquationCalibration

PhysicsMode = Literal["none", "soft_physics", "residual"]


def _reduce_physics_to_total(physics: torch.Tensor, n_joints: int) -> torch.Tensor:
    """Sum decomposed physics components into total torque.

    Handles 3 layouts produced by the dataloader / preprocessor:
      n_joints      — already total torque, returned as-is.
      3 * n_joints  — legacy [g, dyn, f] layout.
      4 * n_joints  — full decomposition [tau_g, tau_M, tau_C, tau_f].
    """
    n = physics.shape[-1]
    if n == n_joints:
        return physics
    nj = n_joints
    if n == 4 * nj:
        return (physics[..., :nj] + physics[..., nj:2*nj]
                + physics[..., 2*nj:3*nj] + physics[..., 3*nj:])
    if n == 3 * nj:
        return physics[..., :nj] + physics[..., nj:2*nj] + physics[..., 2*nj:]
    raise ValueError(
        f"Cannot reduce physics last dim {n} to {n_joints} joints. "
        f"Expected {n_joints}, {3*n_joints}, or {4*n_joints}."
    )


class UnifiedTorqueFNN(nn.Module):
    """
    physics_mode:
      none         — MLP on 15 features (black-box).
      soft_physics — same forward; physics only in training loss (train.py).
      residual     — MLP on [features, tau_phys] with softplus(alpha) and delta heads.
    """

    def __init__(
        self,
        n_joints: int = 5,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "silu",
        physics_mode: PhysicsMode = "none",
    ):
        super().__init__()
        self.n_joints = n_joints
        self.physics_mode: PhysicsMode = physics_mode

        if hidden_layers is None:
            hidden_layers = [256, 512, 256]

        self.config = {
            "n_joints":       n_joints,
            "hidden_layers":  hidden_layers,
            "dropout":        dropout,
            "activation":     activation,
            "physics_mode":   physics_mode,
        }

        if physics_mode == "residual":
            in_dim = n_joints * 4
            Act = ACTIVATION_MAP.get(activation, nn.SiLU)
            encoder_layers: list[nn.Module] = []
            prev = in_dim
            for h in hidden_layers:
                encoder_layers += [nn.Linear(prev, h), nn.LayerNorm(h), Act(), nn.Dropout(dropout)]
                prev = h
            self.encoder = nn.Sequential(*encoder_layers)
            enc_out = hidden_layers[-1] if hidden_layers else in_dim
            self.alpha_head = nn.Linear(enc_out, n_joints)
            self.delta_head = nn.Linear(enc_out, n_joints)
            self.net = None  # type: ignore[assignment]
            self._init_residual_weights()
        else:
            self.net = build_mlp(
                in_dim=n_joints * 3,
                hidden_layers=hidden_layers,
                out_dim=n_joints,
                activation=activation,
                dropout=dropout,
            )
            self.encoder = None  # type: ignore[assignment]
            self.alpha_head = None  # type: ignore[assignment]
            self.delta_head = None  # type: ignore[assignment]
            self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_residual_weights(self) -> None:
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.alpha_head.weight)
        # Init so softplus(bias) ≈ 0.5  →  alpha ≈ 0.5 at h=0.
        # Gentler start than alpha=1.0: avoids large loss spike from full physics error.
        nn.init.constant_(self.alpha_head.bias, math.log(math.exp(0.5) - 1.0))
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, features: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        if self.physics_mode == "residual":
            if physics is None:
                raise ValueError("residual mode requires physics (tau_phys) tensor")
            tau_phys = _reduce_physics_to_total(physics, self.n_joints)
            x = torch.cat([features, tau_phys], dim=-1)
            h = self.encoder(x)
            alpha = F.softplus(self.alpha_head(h)) + 1e-3
            delta = self.delta_head(h)
            self._last_alpha = alpha
            return alpha * tau_phys + delta
        return self.net(features)

    def compute_loss(
        self,
        tau_hat: torch.Tensor,
        tau_measured: torch.Tensor,
        tau_physics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Used by train.py for soft_physics path (physics component only scaled externally)."""
        data_loss = F.mse_loss(tau_hat, tau_measured)
        tau_physics_total = _reduce_physics_to_total(tau_physics, self.n_joints)
        tau_physics_eff = self.tau_calib(tau_physics_total)
        physics_loss = F.mse_loss(tau_hat, tau_physics_eff)
        # Calibration regulariser: keep scale ≈ 1, bias ≈ 0 (same as EC-PINN)
        _s = F.softplus(self.tau_calib.raw_scale) + self.tau_calib.eps
        _calib_reg = (_s - 1.0).pow(2).mean() + self.tau_calib.bias.pow(2).mean()
        physics_loss = physics_loss + 0.01 * _calib_reg
        return {
            "data":    data_loss,
            "physics": physics_loss,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BlackBoxFNN(UnifiedTorqueFNN):
    """Arch 1 — physics_mode fixed to none."""

    def __init__(
        self,
        n_joints: int = 5,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super().__init__(
            n_joints=n_joints,
            hidden_layers=hidden_layers,
            dropout=dropout,
            activation=activation,
            physics_mode="none",
        )


class PhysicsRegularizedFNN(UnifiedTorqueFNN):
    """Arch 2 — soft_physics; same forward as BlackBox."""

    def __init__(
        self,
        n_joints: int = 5,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.05,
        activation: str = "silu",
    ):
        super().__init__(
            n_joints=n_joints,
            hidden_layers=hidden_layers,
            dropout=dropout,
            activation=activation,
            physics_mode="soft_physics",
        )
        self.tau_calib = TauEquationCalibration(n_joints)


class ResidualCorrectionFNN(UnifiedTorqueFNN):
    """Arch 5 — residual on tau_phys."""

    def __init__(
        self,
        n_joints: int = 5,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.05,
        activation: str = "tanh",
    ):
        super().__init__(
            n_joints=n_joints,
            hidden_layers=hidden_layers,
            dropout=dropout,
            activation=activation,
            physics_mode="residual",
        )
