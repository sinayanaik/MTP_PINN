"""Torque MLP building blocks and ``nn.Module`` definitions for training."""

from __future__ import annotations

import torch
import torch.nn as nn

from Neural_Networks.loader import ACTIVE_JOINTS

ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


def build_mlp(
    in_dim: int,
    hidden_layers: list[int],
    out_dim: int,
    activation: str,
    dropout: float,
) -> nn.Sequential:
    if activation not in ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation {activation!r}. Valid choices: {sorted(ACTIVATION_MAP)}"
        )
    Act = ACTIVATION_MAP[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_layers:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), Act(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def reduce_physics_to_total(physics: torch.Tensor, n_joints: int | None = None) -> torch.Tensor:
    """Map decomposed normalised physics (…, 4·J) to summed normalised analytical τ (…, J)."""
    j = ACTIVE_JOINTS if n_joints is None else n_joints
    if physics.shape[-1] != 4 * j:
        raise ValueError(f"expected last dim {4 * j}, got {physics.shape[-1]}")
    lead = physics.shape[:-1]
    p = physics.reshape(*lead, 4, j)
    return p.sum(dim=-2)


class BlackBoxFNN(nn.Module):
    """MLP: (q, qd, qdd) → τ̂."""

    def __init__(
        self,
        n_joints: int = ACTIVE_JOINTS,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super().__init__()
        self.n_joints = n_joints
        if hidden_layers is None:
            hidden_layers = [256, 512, 256]
        self.hparams = {
            "n_joints": n_joints,
            "hidden_layers": list(hidden_layers),
            "dropout": dropout,
            "activation": activation,
        }
        self.net = build_mlp(
            in_dim=n_joints * 3,
            hidden_layers=hidden_layers,
            out_dim=n_joints,
            activation=activation,
            dropout=dropout,
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        del physics
        return self.net(features)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PhysicsRegularizedFNN(nn.Module):
    """MLP: [q, qd, qdd, τ_phys_sum] → τ̂, with learnable per-joint RNEA calibration in loss."""

    def __init__(
        self,
        n_joints: int = ACTIVE_JOINTS,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super().__init__()
        self.n_joints = n_joints
        if hidden_layers is None:
            hidden_layers = [256, 512, 256]
        self.hparams = {
            "n_joints": n_joints,
            "hidden_layers": list(hidden_layers),
            "dropout": dropout,
            "activation": activation,
        }
        # Input is kinematics (3·J) + normalised RNEA sum (J) = 4·J
        self.net = build_mlp(
            in_dim=n_joints * 4,
            hidden_layers=hidden_layers,
            out_dim=n_joints,
            activation=activation,
            dropout=dropout,
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Learnable affine calibration applied to τ_ref inside the physics loss.
        # Initialised to identity so epoch-0 behaviour is unchanged.
        self.cal_scale = nn.Parameter(torch.ones(n_joints))
        self.cal_bias  = nn.Parameter(torch.zeros(n_joints))

    def forward(self, features: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        tau_phys = reduce_physics_to_total(physics, self.n_joints)
        return self.net(torch.cat([features, tau_phys], dim=-1))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualCorrectionFNN(nn.Module):
    """τ̂ = τ_phys + Δ([q, qd, qdd, τ_phys_sum]). Correction net sees the physics estimate."""

    def __init__(
        self,
        n_joints: int = ACTIVE_JOINTS,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super().__init__()
        self.n_joints = n_joints
        if hidden_layers is None:
            hidden_layers = [256, 512, 256]
        self.hparams = {
            "n_joints": n_joints,
            "hidden_layers": list(hidden_layers),
            "dropout": dropout,
            "activation": activation,
        }
        # Input is kinematics (3·J) + normalised RNEA sum (J) = 4·J
        self.net = build_mlp(
            in_dim=n_joints * 4,
            hidden_layers=hidden_layers,
            out_dim=n_joints,
            activation=activation,
            dropout=dropout,
        )
        last_linear: nn.Linear | None = None
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                last_linear = m
        # Warm-start: output layer initialised near-zero so Δ ≈ 0 at epoch 0.
        if last_linear is not None:
            with torch.no_grad():
                last_linear.weight.mul_(1e-2)
                if last_linear.bias is not None:
                    last_linear.bias.mul_(1e-2)

    def forward(self, features: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        if physics is None:
            raise ValueError("ResidualCorrectionFNN requires physics (decomposed τ) from the loader.")
        tau_phys = reduce_physics_to_total(physics, self.n_joints)
        feat_aug = torch.cat([features, tau_phys], dim=-1)
        delta = self.net(feat_aug)
        return tau_phys + delta

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
