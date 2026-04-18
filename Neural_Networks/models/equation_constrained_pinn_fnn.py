"""
EquationConstrainedPINNFNN — Category E.1 (True PINN)
Model #9

Same MLP architecture as BlackBoxFNN. Physics enters through the loss as an
EQUATION RESIDUAL — not matching a precomputed target, but enforcing the
dynamics equation itself.

Physics residual: r = τ̂ - [M(q)q̈ + C(q,q̇)q̇ + g(q) + f(q̇)]
Loss: MSE(τ̂, τ_meas) + λ · MSE(r, 0)

CRITICAL DISTINCTION from Physics-Regularized (Model #3):
    Model #3: receives single precomputed τ_physics as a soft target
    Model #9: receives INDIVIDUAL equation components (τ_M, τ_C, τ_g, τ_f)
              and enforces that τ̂ satisfies the dynamics equation itself

Input:  features (batch, 15) — [q(5), qd(5), qdd(5)]
Output: tau_hat  (batch, 5)

Math:   (q, q̇, q̈) → FNN → τ̂
Loss:   MSE(τ̂, τ_meas) + λ · MSE(τ̂ - [τ_M + τ_C + τ_g + τ_f], 0)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Neural_Networks.models.common import build_mlp
from Neural_Networks.models.tau_equation_calibration import TauEquationCalibration


class EquationConstrainedPINNFNN(nn.Module):
    """
    Equation-Constrained PINN — Feedforward.

    The network predicts tau_hat. The physics loss enforces the dynamics
    equation by penalizing the residual:
        r = tau_hat - (tau_M + tau_C + tau_g + tau_f)

    Unlike Physics-Regularized (Model #3) which treats tau_physics as a
    single opaque target, this model receives individual equation components
    and enforces the STRUCTURE of the dynamics equation itself.
    """

    def __init__(
        self,
        n_joints: int = 5,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.05,
        activation: str = "silu",
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 512, 256]

        self.n_joints = n_joints
        self.config = {
            "n_joints":       n_joints,
            "hidden_layers":  hidden_layers,
            "dropout":        dropout,
            "activation":     activation,
        }

        self.net = build_mlp(
            in_dim=n_joints * 3,
            hidden_layers=hidden_layers,
            out_dim=n_joints,
            activation=activation,
            dropout=dropout,
        )

        # Learnable φ: scales/bias on summed nominal torque before the residual (distinct from PR).
        self.tau_calib = TauEquationCalibration(n_joints)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (batch, 15) — [q(5), qd(5), qdd(5)]

        Returns
        -------
        tau_hat : (batch, 5)
        """
        return self.net(features)

    def compute_loss(
        self,
        tau_hat: torch.Tensor,
        tau_measured: torch.Tensor,
        tau_M: torch.Tensor,
        tau_C: torch.Tensor,
        tau_g: torch.Tensor,
        tau_f: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        L = MSE(τ̂, τ_meas) + λ · [MSE(r, 0) + 0.1 · calib_reg]

        where r = τ̂ - [M(q)q̈ + C(q,q̇)q̇ + g(q) + f(q̇)]
                = τ̂ - (τ_M + τ_C + τ_g + τ_f)

        CRITICAL DISTINCTION from Physics-Regularized (Model #3):
          Model #3: L_phys = MSE(τ̂, τ_physics)     — match a precomputed target
          Model #9: L_phys = MSE(τ̂ - Σ components, 0) — satisfy the equation itself

        Receiving individual components means:
          - The equation structure (inertia + Coriolis + gravity + friction) is explicit
          - Per-component residuals are available for diagnostics
          - If any component model is updated, the residual reflects it automatically

        calib_reg penalises tau_calib drifting away from identity (s≈1, b≈0), preventing
        it from silently absorbing all physics error.

        Parameters
        ----------
        tau_hat      : (batch, 5) — network-predicted torques
        tau_measured : (batch, 5) — ground truth measured torques
        tau_M        : (batch, 5) — M(q)q̈   (inertia, from nominal model)
        tau_C        : (batch, 5) — C(q,q̇)q̇ (Coriolis, from nominal model)
        tau_g        : (batch, 5) — g(q)     (gravity, from nominal model)
        tau_f        : (batch, 5) — f(q̇)    (friction, from nominal model)

        Returns
        -------
        dict with 'data', 'physics' losses and 'residual' for monitoring
        """
        # Data fidelity loss
        data_loss = F.mse_loss(tau_hat, tau_measured)

        # Equation residual: r = τ̂ - τ_eq_eff(φ),  τ_eq_eff = diag(s)τ_eq + b
        tau_equation = tau_M + tau_C + tau_g + tau_f
        tau_equation = self.tau_calib(tau_equation)
        residual = tau_hat - tau_equation

        # Plain MSE residual loss — scale-sensitive so gradient signal diminishes as
        # the model correctly satisfies the dynamics equation.
        physics_loss = F.mse_loss(residual, torch.zeros_like(residual))

        # tau_calib identity regulariser: penalise s≠1 and b≠0.
        # Prevents tau_calib from absorbing physics error without the network learning
        # to satisfy the equation. Light coefficient (0.01) — with correct target-
        # consistent physics normalisation, tau_calib should naturally stay near
        # identity; a heavy penalty would fight legitimate fine-tuning.
        _s = F.softplus(self.tau_calib.raw_scale) + self.tau_calib.eps
        _calib_reg = (_s - 1.0).pow(2).mean() + self.tau_calib.bias.pow(2).mean()
        physics_loss = physics_loss + 0.01 * _calib_reg

        return {
            "data":     data_loss,
            "physics":  physics_loss,
            "residual": residual.detach(),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)