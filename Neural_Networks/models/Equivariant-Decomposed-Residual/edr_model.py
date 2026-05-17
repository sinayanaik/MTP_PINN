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
        use_friction_qdd: bool = False,
        use_phys_cond:    bool = False,
        coriolis_matrix_form: bool = True,
        friction_form:    str = "mlp",
        inertia_psd:      bool = False,
        spectral_norm:    bool = False,
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

        # Physics-conditioned residual: thread the analytic decomposition
        # (τ_g/τ_M/τ_C/τ_f) into the δ-nets component-targeted.  Adds n inputs
        # per network (the relevant decomposition slice).  See
        # ``build_correction_inputs``.
        self._use_phys_cond = bool(use_phys_cond)
        _pc_extra = self.n_joints if self._use_phys_cond else 0
        # Coriolis correction structure: B(q,q̇)·q̇ matrix form (correct,
        # cross-joint) vs legacy element-wise q̇⊙MLP.  See CoriolisCorrection.
        self._coriolis_matrix_form = bool(coriolis_matrix_form)
        # Friction correction structure: "mlp" (legacy) or "stribeck"
        # (structured Coulomb+Stribeck+viscous).  See FrictionCorrection.
        self._friction_form = str(friction_form)
        # δM = L Lᵀ (symmetric + PSD) when True; unconstrained-symmetric when
        # False (default — can also reduce over-estimated nominal inertia).
        self._inertia_psd = bool(inertia_psd)
        # Spectral-norm (Lipschitz cap) on δ-net HIDDEN layers — principled
        # generalisation regulariser; structural guarantees & near-zero init
        # are preserved (final layer is never spectral-normed).
        self._spectral_norm = bool(spectral_norm)

        # Velocity-product dimensionality (upper-triangular entries of q̇⊗q̇).
        # These encode Christoffel-symbol structure τ_C,i = Σ_jk Γ_ijk(q)·q̇_j·q̇_k.
        _vel_prod_dim = self.n_joints * (self.n_joints + 1) // 2
        # Gravity input dim: n_joints (plain) or 3*n_joints (q + sin + cos),
        # plus τ_g (n) when physics-conditioned.
        _gravity_in_dim = (
            (self.n_joints * 3 if self._has_trig_features else self.n_joints)
            + _pc_extra
        )
        # Inertia input dim: gravity config dependence + τ_M (n) when phys-cond.
        _inertia_in_dim = (
            (self.n_joints * 3 if self._has_trig_features else self.n_joints)
            + _pc_extra
        )
        # Coriolis input dim: [q, qd, vel_prod] (+ trig) + τ_C (n) when phys-cond.
        _coriolis_in_dim = (
            (self.n_joints * 4 if self._has_trig_features else self.n_joints * 2)
            + _vel_prod_dim
            + _pc_extra
        )
        # Cached index pairs for extracting upper-triangular q̇⊗q̇ entries.
        # These are private implementation details; external callers should use
        # ``build_correction_inputs`` instead of accessing them directly.
        _up_i, _up_j = torch.triu_indices(self.n_joints, self.n_joints)
        self.register_buffer("_vprod_i", _up_i, persistent=False)
        self.register_buffer("_vprod_j", _up_j, persistent=False)

        # Whether friction correction conditions on |q̈| as well as |q̇|.
        self._use_friction_qdd = bool(use_friction_qdd)

        # Store configuration for checkpointing / analysis tools.
        self.hparams: dict = {
            "n_joints":        self.n_joints,
            "gravity_hidden":  list(gravity_hidden),
            "inertia_hidden":  list(inertia_hidden),
            "inertia_in_dim":  _inertia_in_dim,
            "coriolis_hidden": list(coriolis_hidden),
            "friction_hidden": list(friction_hidden),
            "activation":      activation,
            "correction_dropout": correction_dropout,
            "use_trig_features":  self._has_trig_features,
            "use_friction_qdd":   self._use_friction_qdd,
            "use_phys_cond":      self._use_phys_cond,
            "coriolis_matrix_form": self._coriolis_matrix_form,
            "friction_form":      self._friction_form,
            "inertia_psd":        self._inertia_psd,
            "spectral_norm":      self._spectral_norm,
        }

        # ── Four structurally-constrained correction networks ──────────────
        self.gravity_net  = GravityCorrection(
            n_joints=self.n_joints,
            in_dim=_gravity_in_dim,
            hidden_sizes=gravity_hidden,
            activation=activation,
            dropout=correction_dropout,
            spectral_norm=self._spectral_norm,
        )
        self.inertia_net  = InertiaCorrection(
            n_joints=self.n_joints,
            in_dim=_inertia_in_dim,
            hidden_sizes=inertia_hidden,
            activation=activation,
            dropout=correction_dropout,
            psd=self._inertia_psd,
            spectral_norm=self._spectral_norm,
        )
        self.coriolis_net = CoriolisCorrection(
            n_joints=self.n_joints,
            in_dim=_coriolis_in_dim,
            hidden_sizes=coriolis_hidden,
            activation=activation,
            dropout=correction_dropout,
            matrix_form=self._coriolis_matrix_form,
            spectral_norm=self._spectral_norm,
        )
        self.friction_net = FrictionCorrection(
            n_joints=self.n_joints,
            hidden_sizes=friction_hidden,
            activation=activation,
            dropout=correction_dropout,
            use_qdd=self._use_friction_qdd,
            use_phys_cond=self._use_phys_cond,
            friction_form=self._friction_form,
            spectral_norm=self._spectral_norm,
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
        physics: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build all feature tensors consumed by the correction networks.

        Single source of truth for the feature contract between EDRModel and
        external code (e.g. the training loop and passivity-loss helper).

        Physics-conditioned residual (``use_phys_cond``)
        -----------------------------------------------
        When the model was constructed with ``use_phys_cond=True`` *and* a
        ``physics`` tensor is supplied, the analytic Pinocchio decomposition is
        threaded into each δ-net **component-targeted** (δg sees τ_g, δM sees
        τ_M, δC sees τ_C, δτ_f sees |τ_f|).  This closes the informational gap
        with the PhysReg baseline (which ingests the full decomposition) while
        preserving the decomposed inductive bias — strictly more structured
        than a flat concat.  When ``use_phys_cond=False`` and ``physics=None``
        (passivity helper, model-level tests) the legacy q/q̇-only contract is
        returned unchanged.  When ``use_phys_cond=True`` a physics tensor is
        **required** (a None raises — see B2 guard below).

        Parameters
        ----------
        q:
            Normalised joint positions, shape (B, n_joints).
        qd:
            Normalised joint velocities, shape (B, n_joints).
        physics:
            Optional normalised decomposition [τ_g|τ_M|τ_C|τ_f], shape
            (B, 4·n_joints).  Used only when ``use_phys_cond`` is enabled.

        Returns
        -------
        dict with keys:
            ``gravity_input``   : feature tensor for self.gravity_net.
            ``inertia_input``   : feature tensor for self.inertia_net
                                  (== ``gravity_input`` unless phys-conditioned).
            ``coriolis_input``  : feature tensor for self.coriolis_net.
            ``velocity_products``: upper-triangular entries of q̇⊗q̇, shape (B, n(n+1)/2).
            ``q_raw``           : reconstructed raw joint angles if trig features
                                  are enabled, else None.
        """
        # ── Guards at the source (B2) ────────────────────────────────────
        # Catch shape mistakes here (the vel-product fancy-indexing would
        # otherwise silently mis-broadcast or crash deep in a δ-net).
        n = self.n_joints
        if q.shape[-1] != n or qd.shape[-1] != n:
            raise ValueError(
                f"[EDRModel.build_correction_inputs] q/qd last dim must be "
                f"n_joints={n}, got q={tuple(q.shape)}, qd={tuple(qd.shape)}"
            )
        if self._use_phys_cond and physics is None:
            # Kill the latent trap at the root: under phys-conditioning the
            # δ-nets were built expecting the appended component slices, so a
            # physics=None call would feed wrong-dim inputs.  Fail loudly
            # here rather than crashing deep in a correction net (this also
            # makes the passivity-helper incompatibility explicit at source).
            raise ValueError(
                "[EDRModel.build_correction_inputs] use_phys_cond=True but "
                "physics=None — the δ-net inputs require the decomposition "
                "slices. Pass the physics tensor (or build the model with "
                "use_phys_cond=False for the legacy q/q̇-only contract)."
            )
        vel_prod = qd[:, self._vprod_i] * qd[:, self._vprod_j]
        if self._has_trig_features:
            q_raw = q * self.q_std + self.q_mean
            sin_q = torch.sin(q_raw)
            cos_q = torch.cos(q_raw)
            gravity_base = torch.cat([q, sin_q, cos_q], dim=-1)
            coriolis_base = torch.cat([q, sin_q, cos_q, qd, vel_prod], dim=-1)
        else:
            q_raw = None
            gravity_base = q
            coriolis_base = torch.cat([q, qd, vel_prod], dim=-1)
        # δM(q) shares the gravity configuration dependence (function of q only).
        inertia_base = gravity_base

        if self._use_phys_cond and physics is not None:
            tau_g = physics[:, _PHYS_G_START: _PHYS_G_END]
            tau_M = physics[:, _PHYS_M_START: _PHYS_M_END]
            tau_C = physics[:, _PHYS_C_START: _PHYS_C_END]
            gravity_input  = torch.cat([gravity_base,  tau_g], dim=-1)
            inertia_input  = torch.cat([inertia_base,  tau_M], dim=-1)
            coriolis_input = torch.cat([coriolis_base, tau_C], dim=-1)
        else:
            gravity_input  = gravity_base
            inertia_input  = inertia_base
            coriolis_input = coriolis_base
        return {
            "gravity_input":      gravity_input,
            "inertia_input":      inertia_input,
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

    def compute_corrections(
        self,
        features: torch.Tensor,
        physics:  torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """SINGLE SOURCE OF TRUTH for the EDR computation.

        Validates inputs, builds the correction-network inputs, computes the
        four structured δ-terms, and assembles τ̂.  Both :meth:`forward` and
        the EDR training loop call **this** method, so the trained loss and
        the eval forward provably compute the *same* function (no drift
        between two hand-mirrored code paths — the prior failure mode).

        Returns
        -------
        (tau_hat, deltas)
            ``tau_hat``: predicted torques (B, n_joints), normalised units.
            ``deltas``: dict with the raw correction tensors the training
            loop needs for the regularisation loss / telemetry —
            ``delta_g``, ``delta_M`` (B,n,n matrix), ``delta_M_qdd``,
            ``delta_C_qd``, ``delta_tau_f``.
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
        inputs = self.build_correction_inputs(q, qd, physics)

        # ── Compute structured corrections ───────────────────────────────────
        # δM is computed as the full (B,n,n) matrix ONCE here, then applied to
        # q̈.  Both the torque contribution (bmm) and the Frobenius reg loss
        # (training loop) use this same matrix — no second forward, no drift.
        delta_g     = self.gravity_net(inputs["gravity_input"])          # (B, n)
        delta_M     = self.inertia_net.compute_delta_M(inputs["inertia_input"])  # (B,n,n)
        delta_M_qdd = torch.bmm(delta_M, qdd.unsqueeze(-1)).squeeze(-1)   # (B, n)
        delta_C_qd  = self.coriolis_net(inputs["coriolis_input"], qd)    # (B, n)
        delta_tau_f = self.friction_net(
            qd,
            qdd if self._use_friction_qdd else None,
            tau_f=tau_f if self._use_phys_cond else None,
        )                                                                # (B, n)

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

        deltas = {
            "delta_g":     delta_g,
            "delta_M":     delta_M,        # (B, n, n) — for Frobenius reg loss
            "delta_M_qdd": delta_M_qdd,
            "delta_C_qd":  delta_C_qd,
            "delta_tau_f": delta_tau_f,
        }
        return tau_hat, deltas

    def forward(
        self,
        features: torch.Tensor,
        physics:  torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict joint torques τ̂ (B, n_joints), normalised units.

        Thin wrapper over :meth:`compute_corrections` (the single source of
        truth) — returns only τ̂.  ``physics`` must not be None: EDR
        decomposes corrections per physics component.
        """
        tau_hat, _ = self.compute_corrections(features, physics)
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
