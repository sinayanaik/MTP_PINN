"""
DecomposedStructuredPINNFNN — Category E.2 (Decomposed Structured PINN)
Model #11

Four sub-networks learn CORRECTIONS to each nominal physics component:
    M̂(q)     = M_nom(q) + ΔM_θ(q)     → SPD via Cholesky
    ĉ(q,q̇)  = c_nom(q,q̇) + Δc_θ(q,q̇)
    ĝ(q)     = g_nom(q) + Δg_θ(q)
    f̂(q̇)    = f_nom(q̇) + Δf_θ(q̇)

    τ̂ = M̂(q)q̈ + ĉ(q,q̇) + ĝ(q) + f̂(q̇)

At initialization (corrections ≈ 0): τ̂ ≈ τ_physics_nom (warm start)

Friction parameterization (dissipative by construction):
    Δf_θ(q̇) = −[softplus(v(q̇)) · q̇ + softplus(c(q̇)) · tanh(q̇/ε)]
    Two heads: viscous (∝ q̇) + Coulomb-like (∝ sign(q̇)).  τ_f · q̇ ≤ 0 always.

5-dim physics (total τ_a only — most common case):
    tau_g_nom = total RNEA torque, all others = 0.
    At init: τ̂ ≈ τ_physics_nom.  Networks learn residuals on top.

Multi-term physics loss:
    L = λ_d·L_data + λ_s·L_SPD + λ_f·L_friction + λ_c·L_correction + λ_n·L_nominal
    Note: L_friction ≈ 0 when using 5-dim physics because the dissipative
    parameterization already guarantees τ_f · q̇ ≤ 0 by construction.
    λ_f remains meaningful with 4-dim physics (tau_f_nom may violate dissipation).
    L_correction penalises large deviations from nominal (Occam's razor).
    L_nominal anchors predictions to nominal physics torque.

Input:  features (batch, 15) — [q(5), qd(5), qdd(5)]
        physics  (batch, K) — K=5 total τ_a, or 15 legacy, or 20 decomposed
Output: (tau_hat, components)

Math:   τ̂ = M̂(q)q̈ + ĉ(q,q̇) + ĝ(q) + f̂(q̇)
Loss:   λ_d·MSE(τ̂,τ_meas) + λ_s·SPD_penalty + λ_f·friction_dissipation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Neural_Networks.models.common import build_mlp
from Neural_Networks.models.tau_equation_calibration import TauEquationCalibration


def unpack_nominal_physics(
    physics: torch.Tensor,
    n_joints: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Map ``physics`` batch tensor to nominal (tau_g, tau_M, tau_C, tau_f).

    Materialised datasets usually store only total RNEA torque (``n_joints`` values).
    Some runs may store legacy 3-block (g, tau_dyn, f) or full 4-block decomposition.

    Returns four tensors, each ``(..., n_joints)``. Unused nominal blocks are zeros.
    """
    nj = n_joints
    n = physics.shape[-1]
    if n == 4 * nj:
        return (
            physics[..., :nj],
            physics[..., nj:2 * nj],
            physics[..., 2 * nj:3 * nj],
            physics[..., 3 * nj:4 * nj],
        )
    if n == 3 * nj:
        tau_g = physics[..., :nj]
        z = torch.zeros_like(tau_g)
        tau_dyn = physics[..., nj:2 * nj]
        tau_f = physics[..., 2 * nj:3 * nj]
        return tau_g, z, tau_dyn, tau_f
    if n == nj:
        z = torch.zeros_like(physics)
        return physics, z, z, z
    raise ValueError(
        f"physics last dimension is {n}; expected {nj} (total torque), "
        f"{3 * nj} (legacy g/dyn/f), or {4 * nj} (full decomposition)."
    )


class DecomposedStructuredPINNFNN(nn.Module):
    """
    Decomposed Structured PINN — Feedforward.

    Learns four independent torque component CORRECTIONS with physics-inspired structure:
      - M_net(q)       → SPD inertia matrix via Cholesky  →  tau_M = M_hat @ qdd
      - c_net(q, qd)   → correction to Coriolis/centrifugal →  tau_C = tau_C_nom + c_net
      - g_net(q)       → correction to gravity               →  tau_g = tau_g_nom + g_net
      - f_net(qd)      → two-head friction correction (viscous + Coulomb), both dissipative
                          delta_f = -(softplus(v)·qd + softplus(c)·tanh(qd/ε))
                          tau_f = tau_f_nom + delta_f,  τ_f · q̇ ≤ 0 always

    Corrections initialized near zero → warm start from nominal physics.
    """

    _EPS_SPD  = 1e-4
    # Friction transition width [rad/s] — matches physics.py FRICTION_EPS for consistent
    # tanh(qd / eps) Coulomb term in the correction model.
    _EPS_FRIC = 0.04

    def __init__(
        self,
        n_joints: int = 5,
        inertia_layers: list[int] | None = None,
        coriolis_layers: list[int] | None = None,
        gravity_layers: list[int] | None = None,
        friction_layers: list[int] | None = None,
        dropout: float = 0.05,
        activation: str = "tanh",
        lambda_data: float = 1.0,
        lambda_spd: float = 0.01,
        lambda_friction: float = 0.01,
        lambda_correction_reg: float = 0.001,
        lambda_nominal_consistency: float = 0.01,
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

        _rows, _cols = torch.tril_indices(n_joints, n_joints)
        self.register_buffer('_tril_rows', _rows)
        self.register_buffer('_tril_cols', _cols)
        self.register_buffer('_diag_idx', torch.arange(n_joints))
        self.register_buffer('_eye',      torch.eye(n_joints).unsqueeze(0))

        # lambda_* are stored for direct model.compute_loss() calls;
        # train.py overrides them via hp["spd_weight"] / hp["friction_weight"].
        self.lambda_data     = lambda_data
        self.lambda_spd      = lambda_spd
        self.lambda_friction = lambda_friction
        self.lambda_correction_reg = lambda_correction_reg
        self.lambda_nominal_consistency = lambda_nominal_consistency

        self.config = {
            "n_joints":        n_joints,
            "inertia_layers":  inertia_layers,
            "coriolis_layers": coriolis_layers,
            "gravity_layers":  gravity_layers,
            "friction_layers": friction_layers,
            "dropout":         dropout,
            "activation":      activation,
            "lambda_data":     lambda_data,
            "lambda_spd":      lambda_spd,
            "lambda_friction": lambda_friction,
            "lambda_correction_reg":       lambda_correction_reg,
            "lambda_nominal_consistency":  lambda_nominal_consistency,
        }

        # M_net: q(5) → Cholesky entries(15) → M_hat SPD
        self.M_net = build_mlp(n_joints,     inertia_layers,  self._n_chol, activation, dropout)
        self.c_net = build_mlp(n_joints * 2, coriolis_layers, n_joints,     activation, dropout)
        self.g_net = build_mlp(n_joints,     gravity_layers,  n_joints,     activation, dropout)
        # f_net outputs 2*n_joints: [viscous_coeff, coulomb_coeff] -> softplus ensures dissipation
        self.f_net = build_mlp(n_joints, friction_layers, n_joints * 2, activation, dropout)

        # Learnable affine calibration for nominal physics anchor
        self.tau_calib = TauEquationCalibration(n_joints)

        self._init_weights()

    def _init_weights(self):
        # Xavier for all sub-network internal layers
        for subnet in [self.M_net, self.c_net, self.g_net, self.f_net]:
            for m in subnet.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # Zero-init last layer of c/g/f nets → corrections ≈ 0 at init.
        for subnet in [self.c_net, self.g_net, self.f_net]:
            for m in reversed(list(subnet.modules())):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
                    nn.init.normal_(m.weight, std=1e-3)
                    break
        # M_net last-layer bias:
        #   diagonal Cholesky positions → -2.0  (softplus≈0.126 → M_diag≈0.016)
        #   off-diagonal positions      →  0.0  (L off-diag = 0 → M stays diagonal)
        # Previous scheme (bias=-10 for ALL 15 entries) gave softplus≈4.5e-5 on the
        # diagonal AND drove off-diagonal L factors toward 0 via the same bias, so
        # tau_M≈0 AND ∂tau_M/∂θ≈0 at init — producing catastrophic ~150k initial
        # losses and ~15 wasted epochs crawling out. bias=0 everywhere (softplus(0)
        # ≈0.693) overshoots. The diagonal-only -2 keeps M small but non-degenerate.
        _diag_positions = {int((i * (i + 1)) // 2 + i) for i in range(self.n_joints)}
        for m in reversed(list(self.M_net.modules())):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                for pos in _diag_positions:
                    m.bias.data[pos] = -2.0
                nn.init.normal_(m.weight, std=1e-3)
                break

    def compute_inertia(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute SPD inertia matrix M_hat(q) via Cholesky parameterization.

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

        L[:, self._diag_idx, self._diag_idx] = (
            F.softplus(L[:, self._diag_idx, self._diag_idx]) + self._EPS_SPD
        )

        M = torch.bmm(L, L.transpose(1, 2))
        M = M + self._EPS_SPD * self._eye.float()
        return M.to(input_dtype)

    def forward(self, features: torch.Tensor, physics: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        features : (batch, 15) — [q(5), qd(5), qdd(5)]
        physics  : (batch, K) — K ∈ {nj, 3·nj, 4·nj} (total / legacy / decomposed)

        Returns
        -------
        tau_hat    : (batch, 5)
        components : dict with M, tau_M, tau_C, tau_g, tau_f, delta_c, delta_g, delta_f
        """
        q   = features[..., :self.n_joints]
        qd  = features[..., self.n_joints:2*self.n_joints]
        qdd = features[..., 2*self.n_joints:3*self.n_joints]

        tau_g_nom, _, tau_C_nom, tau_f_nom = unpack_nominal_physics(
            physics, self.n_joints)

        # SPD inertia matrix (learned via Cholesky — corrects tau_M_nom)
        M     = self.compute_inertia(q)
        tau_M = torch.bmm(M, qdd.unsqueeze(-1)).squeeze(-1)

        # Correction sub-networks (zero-initialized → warm start from nominal)
        delta_c = self.c_net(torch.cat([q, qd], dim=-1))
        delta_g = self.g_net(q)
        # Two-head friction correction — both terms are strictly dissipative (τ_f · q̇ ≤ 0):
        #   viscous:  −softplus(v) · q̇              (proportional to velocity)
        #   Coulomb:  −softplus(c) · tanh(q̇ / ε)   (velocity-sign dependent)
        # Split output: first n_joints = viscous coeff, last n_joints = Coulomb coeff.
        f_out   = self.f_net(qd)
        f_visc  = f_out[..., :self.n_joints]
        f_coul  = f_out[..., self.n_joints:]
        delta_f = -(F.softplus(f_visc) * qd +
                    F.softplus(f_coul) * torch.tanh(qd / self._EPS_FRIC))

        # Corrected components: nominal + learned residual (not tau_dyn hack)
        tau_C = tau_C_nom + delta_c
        tau_g = tau_g_nom + delta_g
        tau_f = tau_f_nom + delta_f

        tau_hat = tau_M + tau_C + tau_g + tau_f

        components = {
            "M":       M,
            "tau_M":   tau_M,
            "tau_C":   tau_C,
            "tau_g":   tau_g,
            "tau_f":   tau_f,
            "delta_c": delta_c,   # Coriolis correction for monitoring
            "delta_g": delta_g,   # Gravity correction for monitoring
            "delta_f": delta_f,   # Friction correction (viscous + Coulomb) for monitoring
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
        components: dict[str, torch.Tensor] | None = None,
        tau_physics_nom: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Multi-term decomposed PINN loss:
            L = λ_d·L_data + λ_s·L_SPD + λ_f·L_friction + λ_c·L_correction + λ_n·L_nominal

        Parameters
        ----------
        external_data_loss : optional scalar from trainer (e.g. joint-weighted MSE).
        components         : dict with delta_c, delta_g, delta_f for correction regularisation.
        tau_physics_nom    : nominal physics torque for consistency anchoring.
        """
        if external_data_loss is not None:
            data_loss = external_data_loss
        else:
            data_loss = F.mse_loss(tau_hat, tau_measured)

        eigenvalues = torch.linalg.eigvalsh(M.float())
        spd_violation = torch.clamp(self._EPS_SPD - eigenvalues, min=0.0)
        spd_loss = (spd_violation ** 2).mean()

        # Dissipation check: tau_f · qd ≤ 0 (friction removes energy, never adds it).
        dissipation_product = tau_f * qd
        friction_violation  = torch.clamp(dissipation_product, min=0.0)
        friction_loss       = friction_violation.mean()

        # ── Correction magnitude regularisation (Occam's razor) ─────────────
        # Penalise large corrections to keep model close to nominal physics.
        # Without this, corrections grow unbounded and the model loses its
        # physics-informed warm start advantage.
        correction_loss = tau_hat.new_zeros(())
        if components is not None and self.lambda_correction_reg > 0:
            _dc = components.get("delta_c")
            _dg = components.get("delta_g")
            _df = components.get("delta_f")
            _parts = [t for t in (_dc, _dg, _df) if t is not None]
            if _parts:
                correction_loss = sum((p ** 2).mean() for p in _parts) / len(_parts)

        # ── Nominal consistency loss (physics anchor) ───────────────────────
        # Soft MSE between τ̂ and the CALIBRATED nominal physics torque.
        # tau_calib corrects systematic RNEA bias (MASS_SCALE, friction
        # constants) with a learnable per-joint affine transform.
        nominal_loss = tau_hat.new_zeros(())
        calib_reg = tau_hat.new_zeros(())
        if tau_physics_nom is not None and self.lambda_nominal_consistency > 0:
            tau_nom_eff = self.tau_calib(tau_physics_nom)
            nominal_loss = F.mse_loss(tau_hat, tau_nom_eff)
            _s = F.softplus(self.tau_calib.raw_scale) + self.tau_calib.eps
            calib_reg = (_s - 1.0).pow(2).mean() + self.tau_calib.bias.pow(2).mean()

        # When the trainer provides external_data_loss it owns all weighting;
        # return RAW (unweighted) loss terms so the trainer applies its own
        # hp-driven weights without double-counting.  No "total" key in this
        # branch: any caller that sums it without the trainer's lambdas would
        # get a silently-wrong number.
        if external_data_loss is not None:
            return {
                "data":       data_loss,
                "spd":        spd_loss,
                "friction":   friction_loss,
                "correction": correction_loss,
                "nominal":    nominal_loss,
                "calib_reg":  calib_reg,
            }

        # Standalone mode (no trainer) — apply internal lambdas
        total_loss = (
            self.lambda_data * data_loss
            + self.lambda_spd * spd_loss
            + self.lambda_friction * friction_loss
            + self.lambda_correction_reg * correction_loss
            + self.lambda_nominal_consistency * nominal_loss
            + 0.01 * calib_reg
        )

        return {
            "total":      total_loss,
            "data":       data_loss,
            "spd":        spd_loss,
            "friction":   friction_loss,
            "correction": correction_loss,
            "nominal":    nominal_loss,
            "calib_reg":  calib_reg,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)