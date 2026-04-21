"""EDRModel — full Equivariant-Decomposed-Residual torque prediction model.

Architecture
------------
Given robot state (q, q̇, q̈) and decomposed nominal physics components from
Pinocchio RNEA + friction identification, the model predicts joint torques as:

    τ̂ = [M(q) + δM(q)] q̈
       + [τ_C(q,q̇) + δC(q,q̇)·q̇]
       + [g(q)    + δg(q)]
       + [τ_f(q̇) + δτ_f(q̇)]

Equivalently expressed as a structured residual on top of nominal physics:

    τ̂ = τ_phys + δM(q)·q̈ + δC(q,q̇)·q̇ + δg(q) + δτ_f(q̇)

where all four δ-networks are zero-initialised so that τ̂ = τ_phys at the
start of training.

Data interface (matches Neural_Networks/loader.py)
--------------------------------------------------
features tensor (B, 15):
    [:, 0:5]   → normalised joint positions q
    [:, 5:10]  → normalised joint velocities q̇
    [:, 10:15] → normalised joint accelerations q̈

physics tensor (B, 20):  [tau_g(5) | tau_M(5) | tau_C(5) | tau_f(5)]
    [:, 0:5]   → normalised gravity torque        τ_g
    [:, 5:10]  → normalised inertia torque        τ_M = M(q)·q̈
    [:, 10:15] → normalised Coriolis torque       τ_C = C(q,q̇)·q̇
    [:, 15:20] → normalised friction torque       τ_f

Two-phase curriculum
--------------------
Phase 1 (epochs 1–phase2_start_epoch):
    Only δg and δτ_f are trained.  δM and δC are frozen.  Gravity and
    friction corrections absorb the largest, easiest residuals first.

Phase 2 (epoch ≥ phase2_start_epoch):
    All four corrections are trainable.  After clearing the dominant gravity
    and friction errors in phase 1, the inertia/Coriolis corrections can
    fine-tune the remaining coupling terms without being misled by the large
    gravity/friction residuals.

NASA Defensive Programming conventions
--------------------------------------
All public methods validate inputs (shape, dtype, finiteness) and outputs
(finiteness) with explicit error messages.  No silent fallbacks.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from edr_corrections import (  # local sibling import
    CoriolisCorrection,
    FrictionCorrection,
    GravityCorrection,
    InertiaCorrection,
    correction_parameter_summary,
)

# ---------------------------------------------------------------------------
# Module-level constants — match Neural_Networks/loader.py
# ---------------------------------------------------------------------------

# Number of active joints on the Kikobot arm.
_N_JOINTS_DEFAULT: int = 5

# Expected feature vector length: [q | q̇ | q̈] = 3 × n_joints.
_FEATURE_DIM: int = _N_JOINTS_DEFAULT * 3   # 15

# Expected physics vector length: [τ_g | τ_M | τ_C | τ_f] = 4 × n_joints.
_PHYSICS_DIM: int = _N_JOINTS_DEFAULT * 4   # 20

# Index slices for the physics vector (column ordering from preprocess_data.py).
_PHYS_G_START, _PHYS_G_END   = 0,  5   # gravity torque
_PHYS_M_START, _PHYS_M_END   = 5,  10  # inertia torque  M(q)·q̈
_PHYS_C_START, _PHYS_C_END   = 10, 15  # Coriolis torque C(q,q̇)·q̇
_PHYS_F_START, _PHYS_F_END   = 15, 20  # friction torque τ_f(q̇)

# Index slices for the feature vector.
_FEAT_Q_START,  _FEAT_Q_END   = 0,  5  # joint positions q
_FEAT_QD_START, _FEAT_QD_END  = 5,  10 # joint velocities q̇
_FEAT_QDD_START,_FEAT_QDD_END = 10, 15 # joint accelerations q̈


# ===========================================================================
# EDRModel
# ===========================================================================

class EDRModel(nn.Module):
    """Equivariant-Decomposed-Residual model for joint torque prediction.

    Each physics component receives a structurally-constrained additive
    correction.  All corrections are initialised to zero so that the model
    starts exactly at the nominal RNEA + friction prediction.

    Parameters
    ----------
    n_joints:
        Number of active joints.  Must match the feature and physics tensor
        layouts produced by the data loader.
    gravity_hidden:
        Hidden layer widths for the gravity correction network δg.
    inertia_hidden:
        Hidden layer widths for the inertia correction network δM.
    coriolis_hidden:
        Hidden layer widths for the Coriolis correction network δC.
    friction_hidden:
        Hidden layer widths for the friction correction network h_φ.
    activation:
        Shared nonlinearity for all four sub-networks (e.g. ``silu``, ``tanh``).
    correction_dropout:
        Dropout probability after each hidden activation in all four δ-MLPs
        (``0.0`` disables).
    """

    def __init__(
        self,
        n_joints:         int = _N_JOINTS_DEFAULT,
        gravity_hidden:   Sequence[int] = (64, 64),
        inertia_hidden:   Sequence[int] = (64, 64),
        coriolis_hidden:  Sequence[int] = (64, 64),
        friction_hidden:  Sequence[int] = (32, 32),
        activation:       str = "silu",
        correction_dropout: float = 0.0,
        q_mean:           Sequence[float] | None = None,
        q_std:            Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if n_joints < 1:
            raise ValueError(f"[EDRModel] n_joints must be ≥ 1, got {n_joints}")

        self.n_joints = int(n_joints)
        if not 0.0 <= float(correction_dropout) < 1.0:
            raise ValueError(
                f"[EDRModel] correction_dropout must be in [0, 1), got {correction_dropout!r}"
            )
        correction_dropout = float(correction_dropout)

        # Normalization stats for reconstructing raw joint angles (sin/cos features).
        # When q_mean/q_std are provided, trig features are enabled and the
        # gravity/Coriolis networks receive sin(q_raw), cos(q_raw) as additional
        # physics-informed inputs.
        if q_mean is not None:
            self.register_buffer("q_mean", torch.tensor(q_mean, dtype=torch.float32))
            self.register_buffer("q_std", torch.tensor(q_std, dtype=torch.float32))
            self._has_trig_features = True
        else:
            self.register_buffer("q_mean", torch.zeros(n_joints))
            self.register_buffer("q_std", torch.ones(n_joints))
            self._has_trig_features = False

        # Velocity-product dimensionality (upper-triangular entries of q̇⊗q̇).
        # These encode Christoffel-symbol structure τ_C,i = Σ_jk Γ_ijk(q)·q̇_j·q̇_k.
        _vel_prod_dim = self.n_joints * (self.n_joints + 1) // 2
        # Gravity input dim: n_joints (plain) or 3*n_joints (q + sin + cos).
        _gravity_in_dim = self.n_joints * 3 if self._has_trig_features else self.n_joints
        # Coriolis input dim: [q, qd, vel_prod] or [q, sin(q), cos(q), qd, vel_prod].
        _coriolis_in_dim = (
            (self.n_joints * 4 if self._has_trig_features else self.n_joints * 2)
            + _vel_prod_dim
        )
        # Cached index pairs for extracting upper-triangular q̇⊗q̇ entries.
        # These are private implementation details; external callers should use
        # ``build_correction_inputs`` instead of accessing them directly.
        _up_i, _up_j = torch.triu_indices(self.n_joints, self.n_joints)
        self.register_buffer("_vprod_i", _up_i, persistent=False)
        self.register_buffer("_vprod_j", _up_j, persistent=False)

        # Store configuration for checkpointing / analysis tools.
        self.hparams: dict = {
            "n_joints":        self.n_joints,
            "gravity_hidden":  list(gravity_hidden),
            "inertia_hidden":  list(inertia_hidden),
            "coriolis_hidden": list(coriolis_hidden),
            "friction_hidden": list(friction_hidden),
            "activation":      activation,
            "correction_dropout": correction_dropout,
            "use_trig_features": self._has_trig_features,
        }

        # ── Four structurally-constrained correction networks ──────────────
        self.gravity_net  = GravityCorrection(
            n_joints=self.n_joints,
            in_dim=_gravity_in_dim,
            hidden_sizes=gravity_hidden,
            activation=activation,
            dropout=correction_dropout,
        )
        self.inertia_net  = InertiaCorrection(
            n_joints=self.n_joints,
            hidden_sizes=inertia_hidden,
            activation=activation,
            dropout=correction_dropout,
        )
        self.coriolis_net = CoriolisCorrection(
            n_joints=self.n_joints,
            in_dim=_coriolis_in_dim,
            hidden_sizes=coriolis_hidden,
            activation=activation,
            dropout=correction_dropout,
        )
        self.friction_net = FrictionCorrection(
            n_joints=self.n_joints,
            hidden_sizes=friction_hidden,
            activation=activation,
            dropout=correction_dropout,
        )

        # Curriculum phase tracker (1 or 2).  Updated by set_phase().
        self._phase: int = 1
        # Rolling history of val_rmse (unnormalised/macro) observed during
        # training.  Populated via ``record_val_rmse()`` by the training loop;
        # consumed by the adaptive phase-2 plateau detector in edr_strategy.
        # Not a torch buffer — this is runtime-only training state.
        self._val_rmse_history: list[float] = []
        # Apply phase-1 frozen state immediately after construction.
        self.set_phase(1)

    # -----------------------------------------------------------------------
    # Public API — feature construction and introspection
    # -----------------------------------------------------------------------

    @property
    def use_trig_features(self) -> bool:
        """Whether this model uses sin(q_raw)/cos(q_raw) as physics-informed features.

        Enabled when q_mean and q_std were provided at construction time.
        """
        return self._has_trig_features

    @property
    def q_normalization(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Joint-position normalization stats (mean, std) if trig features are on.

        Returns None when the model was built without q_mean/q_std — in that
        case the gravity network receives only the normalised q.
        """
        if not self._has_trig_features:
            return None
        return (self.q_mean, self.q_std)

    def build_correction_inputs(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build all feature tensors consumed by the correction networks.

        Single source of truth for the feature contract between EDRModel and
        external code (e.g. the training loop and passivity-loss helper).

        Parameters
        ----------
        q:
            Normalised joint positions, shape (B, n_joints).
        qd:
            Normalised joint velocities, shape (B, n_joints).

        Returns
        -------
        dict with keys:
            ``gravity_input``   : feature tensor for self.gravity_net.
            ``coriolis_input``  : feature tensor for self.coriolis_net.
            ``velocity_products``: upper-triangular entries of q̇⊗q̇, shape (B, n(n+1)/2).
            ``q_raw``           : reconstructed raw joint angles if trig features
                                  are enabled, else None.
        """
        vel_prod = qd[:, self._vprod_i] * qd[:, self._vprod_j]
        if self._has_trig_features:
            q_raw = q * self.q_std + self.q_mean
            sin_q = torch.sin(q_raw)
            cos_q = torch.cos(q_raw)
            gravity_input = torch.cat([q, sin_q, cos_q], dim=-1)
            coriolis_input = torch.cat([q, sin_q, cos_q, qd, vel_prod], dim=-1)
        else:
            q_raw = None
            gravity_input = q
            coriolis_input = torch.cat([q, qd, vel_prod], dim=-1)
        return {
            "gravity_input":      gravity_input,
            "coriolis_input":     coriolis_input,
            "velocity_products":  vel_prod,
            "q_raw":              q_raw,
        }

    # -----------------------------------------------------------------------
    # Two-phase curriculum control
    # -----------------------------------------------------------------------

    def set_phase(self, phase: int) -> None:
        """Switch between curriculum phases.

        Phase 1: Only gravity and friction corrections are trainable.
                 Inertia and Coriolis corrections are frozen.
        Phase 2: All four corrections are trainable.

        The phase-1 freeze prevents δM(q)·q̈ from absorbing large gravity
        residuals early in training — the δM network has enough degrees of
        freedom to partially explain gravity errors if allowed to train freely
        from the start.

        Parameters
        ----------
        phase:
            Must be 1 or 2.

        Raises
        ------
        ValueError
            If ``phase`` is not 1 or 2.
        """
        if phase not in (1, 2):
            raise ValueError(f"[EDRModel.set_phase] phase must be 1 or 2, got {phase!r}")

        self._phase = phase

        if phase == 1:
            # Freeze inertia and Coriolis; activate gravity and friction.
            _set_requires_grad(self.inertia_net,  requires_grad=False)
            _set_requires_grad(self.coriolis_net, requires_grad=False)
            _set_requires_grad(self.gravity_net,  requires_grad=True)
            _set_requires_grad(self.friction_net, requires_grad=True)
        else:
            # Phase 2: unfreeze everything.
            _set_requires_grad(self.inertia_net,  requires_grad=True)
            _set_requires_grad(self.coriolis_net, requires_grad=True)
            _set_requires_grad(self.gravity_net,  requires_grad=True)
            _set_requires_grad(self.friction_net, requires_grad=True)

    @property
    def phase(self) -> int:
        """Current curriculum phase (1 or 2)."""
        return self._phase

    # -----------------------------------------------------------------------
    # Validation-metric history (consumed by adaptive phase-2 plateau detector)
    # -----------------------------------------------------------------------

    def record_val_rmse(self, rmse: float) -> None:
        """Append a validation-RMSE observation to the rolling history.

        The training pipeline calls this after each validation evaluation.
        The history is consumed by the adaptive phase-2 transition logic
        in ``edr_strategy._should_transition_to_phase2``.
        """
        self._val_rmse_history.append(float(rmse))

    @property
    def val_rmse_history(self) -> list[float]:
        """Read-only view of recorded validation RMSEs (in order of observation)."""
        return list(self._val_rmse_history)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        features: torch.Tensor,
        physics:  torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict joint torques from kinematic state and nominal physics.

        Parameters
        ----------
        features:
            Normalised kinematic state, shape (B, 3·n_joints).
            Column order: [q (n) | q̇ (n) | q̈ (n)].
        physics:
            Normalised decomposed RNEA + friction components, shape (B, 4·n_joints).
            Column order: [τ_g (n) | τ_M (n) | τ_C (n) | τ_f (n)].
            Must not be None — EDR requires physics components.

        Returns
        -------
        torch.Tensor
            Predicted torques τ̂, shape (B, n_joints), in normalised units.

        Raises
        ------
        ValueError
            If ``physics`` is None, or if any input fails shape/dtype/finiteness
            checks.
        RuntimeError
            If the assembled output contains non-finite values (internal error).
        """
        # ── Input validation ────────────────────────────────────────────────
        if physics is None:
            raise ValueError(
                "[EDRModel] 'physics' tensor is required — EDR decomposes "
                "corrections per physics component and cannot run without it. "
                "Ensure the DataLoader is configured with normalise=True and "
                "that the pre-processed dataset includes filtered_tau_decomposed.csv."
            )

        _n_feat   = self.n_joints * 3
        _n_phys   = self.n_joints * 4

        if features.shape[-1] != _n_feat:
            raise ValueError(
                f"[EDRModel] features last dim must be {_n_feat} "
                f"(= 3 × n_joints={self.n_joints}), got {features.shape[-1]}."
            )
        if physics.shape[-1] != _n_phys:
            raise ValueError(
                f"[EDRModel] physics last dim must be {_n_phys} "
                f"(= 4 × n_joints={self.n_joints}), got {physics.shape[-1]}."
            )
        if features.shape[0] != physics.shape[0]:
            raise ValueError(
                f"[EDRModel] Batch size mismatch: features has {features.shape[0]} rows, "
                f"physics has {physics.shape[0]} rows."
            )
        if not features.dtype.is_floating_point:
            raise ValueError(
                f"[EDRModel] features must be floating-point, got dtype {features.dtype}"
            )
        if not physics.dtype.is_floating_point:
            raise ValueError(
                f"[EDRModel] physics must be floating-point, got dtype {physics.dtype}"
            )
        if not torch.isfinite(features).all():
            n_bad = int((~torch.isfinite(features)).sum().item())
            raise ValueError(
                f"[EDRModel] features contains {n_bad} non-finite value(s). "
                "Check upstream preprocessing and normalisation."
            )
        if not torch.isfinite(physics).all():
            n_bad = int((~torch.isfinite(physics)).sum().item())
            raise ValueError(
                f"[EDRModel] physics contains {n_bad} non-finite value(s). "
                "Check that Pinocchio RNEA ran successfully on this trajectory."
            )

        # ── Unpack features: q, q̇, q̈ ────────────────────────────────────────
        n = self.n_joints
        q   = features[:, _FEAT_Q_START  : _FEAT_Q_END  ]   # (B, n)
        qd  = features[:, _FEAT_QD_START : _FEAT_QD_END ]   # (B, n)
        qdd = features[:, _FEAT_QDD_START: _FEAT_QDD_END]   # (B, n)

        # ── Unpack physics: τ_g, τ_M, τ_C, τ_f ─────────────────────────────
        tau_g = physics[:, _PHYS_G_START: _PHYS_G_END]      # (B, n)
        tau_M = physics[:, _PHYS_M_START: _PHYS_M_END]      # (B, n)
        tau_C = physics[:, _PHYS_C_START: _PHYS_C_END]      # (B, n)
        tau_f = physics[:, _PHYS_F_START: _PHYS_F_END]      # (B, n)

        # ── Build all correction-network inputs via the single-source helper ──
        inputs = self.build_correction_inputs(q, qd)

        # ── Compute structured corrections ───────────────────────────────────
        delta_g     = self.gravity_net(inputs["gravity_input"])          # (B, n)
        delta_M_qdd = self.inertia_net(q, qdd)                            # (B, n)
        delta_C_qd  = self.coriolis_net(inputs["coriolis_input"], qd)    # (B, n)
        delta_tau_f = self.friction_net(qd)                               # (B, n)

        # ── Assemble corrected prediction ─────────────────────────────────
        # τ̂ = (τ_g + δg) + (τ_M + δM·q̈) + (τ_C + δC·q̇) + (τ_f + δτ_f)
        tau_hat = (
            (tau_g + delta_g)
            + (tau_M + delta_M_qdd)
            + (tau_C + delta_C_qd)
            + (tau_f + delta_tau_f)
        )

        # ── Output validation ───────────────────────────────────────────────
        if not torch.isfinite(tau_hat).all():
            n_bad = int((~torch.isfinite(tau_hat)).sum().item())
            raise RuntimeError(
                f"[EDRModel] Assembled τ̂ contains {n_bad} non-finite value(s). "
                "This is an internal error — please file a bug report."
            )

        return tau_hat

    # -----------------------------------------------------------------------
    # Parameter counting
    # -----------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Total trainable parameter count across all four sub-networks."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_all(self) -> int:
        """Total parameter count including frozen parameters."""
        return sum(p.numel() for p in self.parameters())

    def count_parameters_by_component(self) -> dict[str, int]:
        """Per-component trainable parameter counts and grand total.

        Returns
        -------
        dict[str, int]
            Keys: 'gravity', 'inertia', 'coriolis', 'friction', 'total'.
            Note: 'total' counts *all* parameters (including frozen) for an
            honest model-size comparison.
        """
        return correction_parameter_summary(
            gravity=self.gravity_net,
            inertia=self.inertia_net,
            coriolis=self.coriolis_net,
            friction=self.friction_net,
        )

    def __repr__(self) -> str:
        comp = self.count_parameters_by_component()
        lines = [
            f"EDRModel(n_joints={self.n_joints}, phase={self._phase})",
            f"  gravity_net:  {comp['gravity']:>6,} params   δg(q)",
            f"  inertia_net:  {comp['inertia']:>6,} params   δM(q)·q̈   [(A+A^T)/2 symmetric]",
            f"  coriolis_net: {comp['coriolis']:>6,} params   δC(q,q̇)·q̇  [quadratic in ‖q̇‖]",
            f"  friction_net: {comp['friction']:>6,} params   q̇⊙h(|q̇|)   [odd function]",
            f"  {'─' * 42}",
            f"  total:        {comp['total']:>6,} params",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Enable or disable gradient computation for all parameters in a module.

    Parameters
    ----------
    module:
        The nn.Module whose parameters to modify.
    requires_grad:
        True to enable gradient tracking, False to freeze.
    """
    for param in module.parameters():
        param.requires_grad_(requires_grad)
