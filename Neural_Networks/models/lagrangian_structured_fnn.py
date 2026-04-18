"""
LagrangianStructuredFNN — Category D (Lagrangian Structured PINN)
Model #7

Physics-structured network that learns the four torque components separately:
    τ̂ = M̂(q)q̈ + ĉ(q,q̇) + ĝ(q) + f̂(q̇)

- M_net: q → lower-triangular Cholesky factor L → M = LL^T + εI  (SPD guaranteed)
- C_net: [q, q̇] → τ_C   (Coriolis/centrifugal term)
- g_net: q → ��_g          (gravity term)
- f_net: q̇ �� τ_f          (friction term — dissipative by construction)

Physics constraints enforced:
  1. SPD inertia via Cholesky parameterization (architectural)
  2. Dissipative friction via softplus coefficients (architectural)
  3. SPD eigenvalue penalty (loss term)
  4. Friction dissipation penalty (loss term)

Input:  features (batch, 15) — [q(5), qd(5), qdd(5)]
Output: (tau_hat, components) where components is a dict with M, tau_M, tau_C, tau_g, tau_f

Math:   τ̂ = M̂(q)q̈ + ĉ(q,q̇) + ĝ(q) + f̂(q��)
Loss:   MSE(τ̂, τ_measured) + λ_s·SPD_penalty + λ_f·friction_dissipation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Neural_Networks.models.common import build_mlp
from Neural_Networks.models.tau_equation_calibration import TauEquationCalibration


class LagrangianStructuredFNN(nn.Module):
    """
    Lagrangian-structured feedforward network with physics constraints.

    The inertia matrix M(q) is guaranteed SPD via Cholesky parameterization:
        L = lower triangular matrix (diagonal entries passed through softplus for positivity)
        M = L @ L.T + eps * I

    Friction is guaranteed dissipative by construction:
        f(q̇) = −[softplus(v) · q̇ + softplus(c) · tanh(q̇/ε)]
        → τ_f · q̇ ≤ 0 always (friction removes energy, never adds it)

    For n_joints=5: the lower-triangular Cholesky factor has 5*(5+1)/2 = 15 entries.
    """

    _EPS_SPD  = 1e-4
    _EPS_FRIC = 0.04   # Friction transition width [rad/s], matches physics.py

    def __init__(
        self,
        n_joints: int = 5,
        inertia_layers: list[int] | None = None,
        coriolis_layers: list[int] | None = None,
        gravity_layers: list[int] | None = None,
        friction_layers: list[int] | None = None,
        dropout: float = 0.05,
        activation: str = "tanh",
        lambda_spd: float = 0.01,
        lambda_friction: float = 0.01,
    ):
        super().__init__()
        if inertia_layers is None:
            inertia_layers = [256, 512, 256]
        if coriolis_layers is None:
            coriolis_layers = [256, 512, 256]
        if gravity_layers is None:
            gravity_layers = [256, 512, 256]
        if friction_layers is None:
            friction_layers = [128, 128]

        self.n_joints = n_joints
        self._n_chol = n_joints * (n_joints + 1) // 2
        self.lambda_spd = lambda_spd
        self.lambda_friction = lambda_friction

        _rows, _cols = torch.tril_indices(n_joints, n_joints)
        self.register_buffer('_tril_rows', _rows)
        self.register_buffer('_tril_cols', _cols)
        self.register_buffer('_diag_idx', torch.arange(n_joints))
        self.register_buffer('_eye', torch.eye(n_joints).unsqueeze(0))

        self.config = {
            "n_joints":        n_joints,
            "inertia_layers":  inertia_layers,
            "coriolis_layers": coriolis_layers,
            "gravity_layers":  gravity_layers,
            "friction_layers": friction_layers,
            "dropout":         dropout,
            "activation":      activation,
            "lambda_spd":      lambda_spd,
            "lambda_friction": lambda_friction,
        }

        # M_net: q(5) → Cholesky entries (15 for 5-joint robot)
        self.M_net = build_mlp(n_joints, inertia_layers, self._n_chol, activation, dropout)

        # C_net: [q(5), qd(5)] → tau_C(5)
        self.C_net = build_mlp(n_joints * 2, coriolis_layers, n_joints, activation, dropout)

        # g_net: q(5) → tau_g(5)
        self.g_net = build_mlp(n_joints, gravity_layers, n_joints, activation, dropout)

        # Learnable affine calibration for nominal physics anchor
        self.tau_calib = TauEquationCalibration(n_joints)

        # f_net: qd(5) → 2*n_joints (viscous + Coulomb coefficients)
        # Dissipative friction: −[softplus(v)·q̇ + softplus(c)·tanh(q̇/ε)]
        self.f_net = build_mlp(n_joints, friction_layers, n_joints * 2, activation, dropout)

        self._init_weights()

    def _init_weights(self):
        for subnet in [self.M_net, self.C_net, self.g_net, self.f_net]:
            for m in subnet.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # M_net last-layer bias: diagonal entries → -2.0 (softplus≈0.126) so
        # M ≈ 0.016·I at init; off-diagonal entries → 0 so L is diagonal and
        # M stays near-isotropic. Previous bias=-10 for all 15 entries gave
        # softplus≈4.5e-5 which made τ_M≈0 AND τ-deriv≈0, causing catastrophic
        # ~150k initial losses and ~15 wasted epochs before the model escaped.
        _diag_positions = {int((i * (i + 1)) // 2 + i) for i in range(self.n_joints)}
        for m in reversed(list(self.M_net.modules())):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                for pos in _diag_positions:
                    m.bias.data[pos] = -2.0
                nn.init.normal_(m.weight, std=1e-3)
                break

        # f_net last layer: small weights + zero bias → friction ≈ 0 at init
        for m in reversed(list(self.f_net.modules())):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.zeros_(m.bias)
                break

    def compute_inertia(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute SPD inertia matrix M(q) via Cholesky parameterization.

        Parameters
        ----------
        q : (batch, n_joints)

        Returns
        -------
        M : (batch, n_joints, n_joints) — symmetric positive definite
        """
        batch = q.shape[0]
        input_dtype = q.dtype
        # Compute entirely in float32 for numerical stability (AMP-safe: F.softplus always returns float32)
        chol_entries = self.M_net(q).float()

        L = torch.zeros(batch, self.n_joints, self.n_joints,
                        device=q.device, dtype=torch.float32)
        L[:, self._tril_rows, self._tril_cols] = chol_entries

        # Ensure diagonal entries are strictly positive via softplus
        L[:, self._diag_idx, self._diag_idx] = (
            F.softplus(L[:, self._diag_idx, self._diag_idx]) + self._EPS_SPD
        )

        # M = L @ L^T + eps * I  (SPD guaranteed)
        M = torch.bmm(L, L.transpose(1, 2))
        M = M + self._EPS_SPD * self._eye.float()
        return M.to(input_dtype)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        features : (batch, 15) — [q(5), qd(5), qdd(5)]

        Returns
        -------
        tau_hat    : (batch, 5)
        components : dict with keys M, tau_M, tau_C, tau_g, tau_f
        """
        q   = features[..., :self.n_joints]                    # (B, 5)
        qd  = features[..., self.n_joints:2*self.n_joints]     # (B, 5)
        qdd = features[..., 2*self.n_joints:3*self.n_joints]   # (B, 5)

        M     = self.compute_inertia(q)                        # (B, 5, 5) SPD
        tau_M = torch.bmm(M, qdd.unsqueeze(-1)).squeeze(-1)    # (B, 5)
        tau_C = self.C_net(torch.cat([q, qd], dim=-1))         # (B, 5)
        tau_g = self.g_net(q)                                   # (B, 5)

        # Dissipative friction: guaranteed τ_f · q̇ ≤ 0
        f_out  = self.f_net(qd)
        f_visc = f_out[..., :self.n_joints]
        f_coul = f_out[..., self.n_joints:]
        tau_f  = -(F.softplus(f_visc) * qd +
                   F.softplus(f_coul) * torch.tanh(qd / self._EPS_FRIC))

        tau_hat = tau_M + tau_C + tau_g + tau_f

        components = {
            "M":     M,
            "tau_M": tau_M,
            "tau_C": tau_C,
            "tau_g": tau_g,
            "tau_f": tau_f,
        }
        return tau_hat, components

    def compute_loss(
        self,
        tau_hat: torch.Tensor,
        tau_measured: torch.Tensor,
        qd: torch.Tensor,
        M: torch.Tensor,
        tau_f: torch.Tensor,
        external_data_loss: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Physics constraint losses for Lagrangian structure.

            L = λ_d·L_data + λ_s·L_SPD + ��_f·L_friction

        SPD loss:      penalises eigenvalues below ε (reinforces Cholesky guarantee)
        Friction loss: penalises τ_f · q̇ > 0 (reinforces dissipative construction)

        Both constraints are ALREADY satisfied architecturally, but the loss
        terms provide gradient signal to keep the model well-conditioned
        during training (prevents numerical edge cases at extreme inputs).
        """
        if external_data_loss is not None:
            data_loss = external_data_loss
        else:
            data_loss = F.mse_loss(tau_hat, tau_measured)

        # SPD eigenvalue penalty
        eigenvalues = torch.linalg.eigvalsh(M.float())
        spd_violation = torch.clamp(self._EPS_SPD - eigenvalues, min=0.0)
        spd_loss = (spd_violation ** 2).mean()

        # Dissipation penalty (should be near zero by construction)
        dissipation_product = tau_f * qd
        friction_violation  = torch.clamp(dissipation_product, min=0.0)
        friction_loss       = friction_violation.mean()

        total_loss = (
            data_loss
            + self.lambda_spd * spd_loss
            + self.lambda_friction * friction_loss
        )

        return {
            "total":    total_loss,
            "data":     data_loss,
            "spd":      spd_loss,
            "friction": friction_loss,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
