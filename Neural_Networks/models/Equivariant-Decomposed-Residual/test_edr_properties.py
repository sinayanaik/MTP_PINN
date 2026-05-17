"""Structural property tests for the EDR model.

These tests verify that the mathematical constraints guaranteed by the
architecture hold exactly (to floating-point precision) for every parameter
configuration, not just after training.  They are the first thing to run
before any training experiment.

Tests are deliberately written to be fast (CPU, small batches), deterministic
(fixed seeds), and self-contained (no external data files required).

Usage (from repository root)::

    PYTHONPATH=. pytest Neural_Networks/models/Equivariant-Decomposed-Residual/test_edr_properties.py -v

NASA Defensive Programming note: every test has a clear assertion message that
identifies what property failed and what the measured value was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup — must precede EDR imports.
# ---------------------------------------------------------------------------
_EDR_DIR   = Path(__file__).resolve().parent
_REPO_ROOT = _EDR_DIR.parent.parent.parent

for _p in (str(_EDR_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from edr_corrections import (  # noqa: E402
    CoriolisCorrection,
    FrictionCorrection,
    GravityCorrection,
    InertiaCorrection,
    correction_parameter_summary,
)
from edr_model import EDRModel  # noqa: E402
from edr_strategy import (  # noqa: E402
    _correction_reg_loss,
    _resolve_component_lambdas,
    _resolve_joint_weights,
    _should_transition_to_phase2,
    _validate_edr_hp,
)

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
_DEVICE     = torch.device("cpu")
_SEED       = 42
_N_JOINTS   = 5
_BATCH      = 512   # Larger batch catches statistical outliers more reliably.
_ATOL_EXACT = 1e-5  # Tolerance for "exact by construction" properties.
_ATOL_INIT  = 5e-3  # Near-zero init tolerance (std=1e-4 last layer ⇒ O(1e-3) max).


def _rand(*shape: int, requires_grad: bool = False) -> torch.Tensor:
    """Reproducible random float32 tensor in [-1, 1]."""
    torch.manual_seed(_SEED)
    return (2.0 * torch.rand(*shape) - 1.0).requires_grad_(requires_grad)


def _rand_positive(*shape: int) -> torch.Tensor:
    """Reproducible random float32 tensor in [0.01, 1] (strictly positive)."""
    torch.manual_seed(_SEED)
    return (torch.rand(*shape) * 0.99 + 0.01)


# ===========================================================================
# Individual correction network tests
# ===========================================================================


class TestGravityCorrection:
    """GravityCorrection: basic shape and zero-init tests."""

    def setup_method(self) -> None:
        torch.manual_seed(_SEED)
        self.net = GravityCorrection(n_joints=_N_JOINTS)

    def test_output_shape(self) -> None:
        q = _rand(_BATCH, _N_JOINTS)
        out = self.net(q)
        assert out.shape == (_BATCH, _N_JOINTS), (
            f"GravityCorrection output shape should be ({_BATCH}, {_N_JOINTS}), got {out.shape}"
        )

    def test_zero_init(self) -> None:
        """At construction, δg(q) ≈ 0 for any q (last layer is zero-initialised)."""
        q   = _rand(_BATCH, _N_JOINTS)
        out = self.net(q)
        max_abs = out.abs().max().item()
        assert max_abs < _ATOL_INIT, (
            f"GravityCorrection should output ≈ 0 at initialisation, "
            f"but max |δg| = {max_abs:.2e}"
        )

    def test_output_finite(self) -> None:
        q = _rand(_BATCH, _N_JOINTS)
        out = self.net(q)
        assert torch.isfinite(out).all(), "GravityCorrection output contains NaN/Inf"

    def test_rejects_wrong_last_dim(self) -> None:
        q_bad = _rand(_BATCH, _N_JOINTS + 1)
        with pytest.raises(ValueError, match="last dimension"):
            self.net(q_bad)

    def test_parameter_count_positive(self) -> None:
        assert self.net.count_parameters() > 0, "GravityCorrection has no trainable params"

    def test_augmented_input(self) -> None:
        """GravityCorrection accepts augmented input [q, sin(q), cos(q)]."""
        torch.manual_seed(_SEED)
        net_aug = GravityCorrection(n_joints=_N_JOINTS, in_dim=_N_JOINTS * 3)
        q_aug = _rand(_BATCH, _N_JOINTS * 3)
        out = net_aug(q_aug)
        assert out.shape == (_BATCH, _N_JOINTS), (
            f"Augmented GravityCorrection output shape should be ({_BATCH}, {_N_JOINTS}), got {out.shape}"
        )
        assert torch.isfinite(out).all(), "Augmented GravityCorrection output contains NaN/Inf"


class TestInertiaCorrection:
    """InertiaCorrection: symmetry and zero-init are the critical properties."""

    def setup_method(self) -> None:
        torch.manual_seed(_SEED)
        self.net = InertiaCorrection(n_joints=_N_JOINTS)

    def test_output_shape(self) -> None:
        q   = _rand(_BATCH, _N_JOINTS)
        qdd = _rand(_BATCH, _N_JOINTS)
        out = self.net(q, qdd)
        assert out.shape == (_BATCH, _N_JOINTS), (
            f"InertiaCorrection output shape should be ({_BATCH}, {_N_JOINTS}), got {out.shape}"
        )

    def test_delta_M_symmetric(self) -> None:
        """δM(q) = δM(q)^T exactly for every q and every parameter setting.

        This is the most critical structural property.  It holds by the
        (A + A^T)/2 construction regardless of what values q or the MLP
        parameters take.
        """
        q        = _rand(_BATCH, _N_JOINTS)
        delta_M  = self.net.compute_delta_M(q)        # (B, n, n)
        delta_MT = delta_M.transpose(1, 2)            # (B, n, n)
        max_asym = (delta_M - delta_MT).abs().max().item()
        assert max_asym < _ATOL_EXACT, (
            f"InertiaCorrection: δM should be exactly symmetric ((A+A^T)/2 construction), "
            f"but max |δM - δM^T| = {max_asym:.2e}  (tolerance: {_ATOL_EXACT:.0e})"
        )

    def test_zero_init(self) -> None:
        """At construction, δM(q)·q̈ ≈ 0 (A≈0 gives δM≈0)."""
        q   = _rand(_BATCH, _N_JOINTS)
        qdd = _rand(_BATCH, _N_JOINTS)
        out = self.net(q, qdd)
        max_abs = out.abs().max().item()
        assert max_abs < _ATOL_INIT, (
            f"InertiaCorrection should output ≈ 0 at initialisation, "
            f"but max |δM·q̈| = {max_abs:.2e}"
        )

    def test_delta_M_shape(self) -> None:
        q        = _rand(8, _N_JOINTS)
        delta_M  = self.net.compute_delta_M(q)
        assert delta_M.shape == (8, _N_JOINTS, _N_JOINTS), (
            f"compute_delta_M output shape should be (8, {_N_JOINTS}, {_N_JOINTS}), "
            f"got {delta_M.shape}"
        )

    def test_output_finite(self) -> None:
        q   = _rand(_BATCH, _N_JOINTS)
        qdd = _rand(_BATCH, _N_JOINTS)
        out = self.net(q, qdd)
        assert torch.isfinite(out).all(), "InertiaCorrection output contains NaN/Inf"

    def test_batch_size_mismatch_raises(self) -> None:
        q   = _rand(10, _N_JOINTS)
        qdd = _rand(11, _N_JOINTS)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            self.net(q, qdd)


class TestCoriolisCorrection:
    """CoriolisCorrection: vanishing at zero velocity is the critical property."""

    def setup_method(self) -> None:
        torch.manual_seed(_SEED)
        self.net = CoriolisCorrection(n_joints=_N_JOINTS)

    def _make_features(self, q, qd):
        return torch.cat([q, qd], dim=-1)

    def test_output_shape(self) -> None:
        q  = _rand(_BATCH, _N_JOINTS)
        qd = _rand(_BATCH, _N_JOINTS)
        out = self.net(self._make_features(q, qd), qd)
        assert out.shape == (_BATCH, _N_JOINTS), (
            f"CoriolisCorrection output shape should be ({_BATCH}, {_N_JOINTS}), got {out.shape}"
        )

    def test_vanishes_at_zero_velocity(self) -> None:
        """δC·q̇ = 0 exactly when the *full* q̇ vector is zero.

        Coriolis forces vanish when the robot is stationary.  The matrix
        construction δC·q̇ = B(q,q̇) @ q̇ ensures this: B · 0 = 0 for any B
        (a *vector* property — not the stricter, physically-wrong per-joint
        property of the legacy element-wise form).
        """
        q  = _rand(_BATCH, _N_JOINTS)
        qd = torch.zeros(_BATCH, _N_JOINTS)
        out = self.net(self._make_features(q, qd), qd)
        max_abs = out.abs().max().item()
        assert max_abs < 1e-7, (
            f"CoriolisCorrection must vanish when full q̇ = 0 (B·0=0), "
            f"but max |δC·q̇| = {max_abs:.2e}"
        )

    def test_cross_joint_coupling(self) -> None:
        """Matrix form: joint i's Coriolis ≠ 0 when q̇_i=0 but others move.

        This is the physically-correct behaviour the legacy element-wise
        q̇⊙MLP form structurally forbade.  With a trained-away-from-zero B,
        moving only joint 1 must induce a non-zero correction on some other
        joint.  (Uses a fresh net with non-zero final layer so B ≠ 0.)
        """
        torch.manual_seed(_SEED)
        net = CoriolisCorrection(n_joints=_N_JOINTS, matrix_form=True)
        # Break the near-zero init so B is a generic non-zero matrix.
        with torch.no_grad():
            for p in net.net[-1].parameters():
                p.add_(torch.randn_like(p) * 0.5)
        q  = _rand(_BATCH, _N_JOINTS)
        qd = torch.zeros(_BATCH, _N_JOINTS)
        qd[:, 1] = 0.7  # only joint 1 moves; q̇_0 = q̇_2.. = 0
        out = net(self._make_features(q, qd), qd)
        # Some joint OTHER than joint 1 must receive a non-zero correction.
        other = torch.cat([out[:, :1], out[:, 2:]], dim=1)
        assert other.abs().max().item() > 1e-4, (
            "Matrix-form Coriolis must allow cross-joint coupling: moving only "
            f"joint 1 left all other joints exactly zero (max={other.abs().max().item():.2e})"
        )
        # And it must still vanish when the FULL q̇ vector is zero.
        out0 = net(self._make_features(q, torch.zeros_like(qd)), torch.zeros_like(qd))
        assert out0.abs().max().item() < 1e-6, "B·0 must be exactly 0"

    def test_no_nan_at_zero_velocity(self) -> None:
        """Zero velocity must not produce NaN."""
        q  = _rand(_BATCH, _N_JOINTS)
        qd = torch.zeros(_BATCH, _N_JOINTS)
        out = self.net(self._make_features(q, qd), qd)
        assert torch.isfinite(out).all(), (
            "CoriolisCorrection produced NaN/Inf at q̇ = 0"
        )

    def test_zero_init(self) -> None:
        """At construction, δC(q,q̇)·q̇ ≈ 0 for any inputs."""
        q  = _rand(_BATCH, _N_JOINTS)
        qd = _rand(_BATCH, _N_JOINTS)
        out = self.net(self._make_features(q, qd), qd)
        max_abs = out.abs().max().item()
        assert max_abs < _ATOL_INIT, (
            f"CoriolisCorrection should output ≈ 0 at initialisation, "
            f"but max |δC·q̇| = {max_abs:.2e}"
        )

    def test_output_finite(self) -> None:
        q  = _rand(_BATCH, _N_JOINTS)
        qd = _rand(_BATCH, _N_JOINTS)
        out = self.net(self._make_features(q, qd), qd)
        assert torch.isfinite(out).all(), "CoriolisCorrection output contains NaN/Inf"

    def test_augmented_input(self) -> None:
        """CoriolisCorrection accepts augmented input [q, sin(q), cos(q), qd]."""
        torch.manual_seed(_SEED)
        net_aug = CoriolisCorrection(n_joints=_N_JOINTS, in_dim=_N_JOINTS * 4)
        q = _rand(_BATCH, _N_JOINTS)
        qd = _rand(_BATCH, _N_JOINTS)
        feat = torch.cat([q, torch.sin(q), torch.cos(q), qd], dim=-1)
        out = net_aug(feat, qd)
        assert out.shape == (_BATCH, _N_JOINTS)
        # Must still vanish at qd=0
        qd_zero = torch.zeros(_BATCH, _N_JOINTS)
        feat_zero = torch.cat([q, torch.sin(q), torch.cos(q), qd_zero], dim=-1)
        out_zero = net_aug(feat_zero, qd_zero)
        assert out_zero.abs().max().item() < 1e-7


class TestFrictionCorrection:
    """FrictionCorrection: odd symmetry is the critical property."""

    def setup_method(self) -> None:
        torch.manual_seed(_SEED)
        self.net = FrictionCorrection(n_joints=_N_JOINTS)

    def test_output_shape(self) -> None:
        qd  = _rand(_BATCH, _N_JOINTS)
        out = self.net(qd)
        assert out.shape == (_BATCH, _N_JOINTS), (
            f"FrictionCorrection output shape should be ({_BATCH}, {_N_JOINTS}), got {out.shape}"
        )

    def test_odd_function_symmetry(self) -> None:
        """δτ_f(−q̇) = −δτ_f(q̇) exactly for every q̇ and every parameter setting.

        This is the most critical structural property.  It holds by the
        product trick: q̇ ⊙ h(|q̇|) — the sign of q̇ carries the odd symmetry,
        |q̇| is even, and the product is exactly odd.
        """
        qd      = _rand(_BATCH, _N_JOINTS)
        out_pos = self.net(qd)
        out_neg = self.net(-qd)
        max_violation = (out_pos + out_neg).abs().max().item()
        assert max_violation < _ATOL_EXACT, (
            f"FrictionCorrection: δτ_f(q̇) + δτ_f(−q̇) should be exactly 0 "
            f"(odd-function construction), but max violation = {max_violation:.2e}  "
            f"(tolerance: {_ATOL_EXACT:.0e})"
        )

    def test_zero_init(self) -> None:
        """At construction, δτ_f(q̇) ≈ 0 for any q̇."""
        qd  = _rand(_BATCH, _N_JOINTS)
        out = self.net(qd)
        max_abs = out.abs().max().item()
        assert max_abs < _ATOL_INIT, (
            f"FrictionCorrection should output ≈ 0 at initialisation, "
            f"but max |δτ_f| = {max_abs:.2e}"
        )

    def test_zero_velocity_output(self) -> None:
        """δτ_f(0) = 0 · h(0) = 0 exactly (product trick consequence)."""
        qd  = torch.zeros(_BATCH, _N_JOINTS)
        out = self.net(qd)
        max_abs = out.abs().max().item()
        assert max_abs < 1e-10, (
            f"FrictionCorrection at q̇=0: expected exactly 0, got max |δτ_f| = {max_abs:.2e}"
        )

    def test_output_finite(self) -> None:
        qd  = _rand(_BATCH, _N_JOINTS)
        out = self.net(qd)
        assert torch.isfinite(out).all(), "FrictionCorrection output contains NaN/Inf"

    def test_rejects_wrong_last_dim(self) -> None:
        qd_bad = _rand(_BATCH, _N_JOINTS + 2)
        with pytest.raises(ValueError, match="last dimension"):
            self.net(qd_bad)


class TestCorrectionParameterSummary:
    """Sanity-check the parameter counting utility."""

    def test_summary_keys(self) -> None:
        g = GravityCorrection(_N_JOINTS)
        m = InertiaCorrection(_N_JOINTS)
        c = CoriolisCorrection(_N_JOINTS)
        f = FrictionCorrection(_N_JOINTS)
        summary = correction_parameter_summary(g, m, c, f)
        assert set(summary.keys()) == {"gravity", "inertia", "coriolis", "friction", "total"}, (
            f"Unexpected keys in parameter summary: {set(summary.keys())}"
        )

    def test_total_equals_sum(self) -> None:
        g = GravityCorrection(_N_JOINTS)
        m = InertiaCorrection(_N_JOINTS)
        c = CoriolisCorrection(_N_JOINTS)
        f = FrictionCorrection(_N_JOINTS)
        summary = correction_parameter_summary(g, m, c, f)
        expected_total = summary["gravity"] + summary["inertia"] + summary["coriolis"] + summary["friction"]
        assert summary["total"] == expected_total, (
            f"Parameter summary 'total' ({summary['total']}) != sum of components ({expected_total})"
        )

    def test_total_approximately_1800(self) -> None:
        """Sanity-check the parameter count is in a reasonable range.

        Small hidden widths [32,32]/[16,16] stay in the low-thousands regime
        (still negligible vs a 200k+ MLP baseline).
        """
        g = GravityCorrection(_N_JOINTS,  hidden_sizes=(32, 32))
        m = InertiaCorrection(_N_JOINTS,  hidden_sizes=(32, 32))
        c = CoriolisCorrection(_N_JOINTS, hidden_sizes=(32, 32))
        f = FrictionCorrection(_N_JOINTS, hidden_sizes=(16, 16))
        summary = correction_parameter_summary(g, m, c, f)
        assert 500 < summary["total"] < 20_000, (
            f"Total parameter count {summary['total']} is unexpectedly large or small. "
            "Expected 500–20,000 for [32,32]/[16,16] hidden sizes."
        )


class TestGravityCorrectionDropout:
    """Dropout on correction MLPs: forward is finite in train mode."""

    def setup_method(self) -> None:
        torch.manual_seed(_SEED)
        self.net = GravityCorrection(n_joints=_N_JOINTS, hidden_sizes=(32, 32), dropout=0.15)

    def test_train_mode_finite(self) -> None:
        self.net.train()
        q = _rand(_BATCH, _N_JOINTS)
        out = self.net(q)
        assert torch.isfinite(out).all(), "GravityCorrection with dropout produced NaN/Inf"


class TestEDRModelDropoutForward:
    """Assembled model with correction_dropout: train forward stays finite."""

    def test_train_forward_finite(self) -> None:
        torch.manual_seed(_SEED)
        m = EDRModel(
            n_joints=_N_JOINTS,
            gravity_hidden=(32, 32),
            inertia_hidden=(32, 32),
            coriolis_hidden=(32, 32),
            friction_hidden=(16, 16),
            correction_dropout=0.12,
        )
        m.train()
        feat = _rand(_BATCH, _N_JOINTS * 3)
        phys = _rand(_BATCH, _N_JOINTS * 4)
        out = m(feat, phys)
        assert torch.isfinite(out).all(), "EDRModel with dropout produced NaN/Inf in train mode"


# ===========================================================================
# Full EDRModel tests
# ===========================================================================


class TestEDRModelProperties:
    """Tests on the assembled EDRModel."""

    def setup_method(self) -> None:
        torch.manual_seed(_SEED)
        self.model = EDRModel(n_joints=_N_JOINTS)
        self.model.eval()

    def test_default_param_scale(self) -> None:
        """Default EDRModel is wider than the guide’s ~1.8k quote but still ≪ BlackBoxFNN."""
        comp = self.model.count_parameters_by_component()
        total = comp["total"]
        assert 5_000 < total < 200_000, (
            f"Default EDRModel total params {total} outside expected band 5k–200k."
        )

    def _make_features(self, B: int = _BATCH) -> torch.Tensor:
        return _rand(B, _N_JOINTS * 3)

    def _make_physics(self, B: int = _BATCH) -> torch.Tensor:
        return _rand(B, _N_JOINTS * 4)

    # ── Shape tests ─────────────────────────────────────────────────────────

    def test_output_shape(self) -> None:
        feat  = self._make_features()
        phys  = self._make_physics()
        out   = self.model(feat, phys)
        assert out.shape == (_BATCH, _N_JOINTS), (
            f"EDRModel output shape should be ({_BATCH}, {_N_JOINTS}), got {out.shape}"
        )

    def test_output_finite(self) -> None:
        feat = self._make_features()
        phys = self._make_physics()
        out  = self.model(feat, phys)
        assert torch.isfinite(out).all(), "EDRModel forward output contains NaN/Inf"

    # ── Zero-initialisation test ─────────────────────────────────────────────

    def test_starts_at_physics(self) -> None:
        """At initialisation, τ̂ = τ_phys exactly (all δ-nets output zero).

        This is the critical safety property of the EDR design.  At epoch 0
        the model makes identical predictions to the nominal physics model.
        Training can only improve from there.
        """
        feat = self._make_features()
        phys = self._make_physics()

        tau_hat  = self.model(feat, phys)
        # Sum all four physics components to get the nominal prediction.
        n = _N_JOINTS
        tau_phys = (
            phys[:, 0:n]
            + phys[:, n:2*n]
            + phys[:, 2*n:3*n]
            + phys[:, 3*n:4*n]
        )

        max_diff = (tau_hat - tau_phys).abs().max().item()
        assert max_diff < _ATOL_INIT, (
            f"EDRModel at initialisation: τ̂ should equal τ_phys (all δ=0), "
            f"but max |τ̂ − τ_phys| = {max_diff:.2e}  (tolerance: {_ATOL_INIT:.0e})"
        )

    # ── Physics-None raises ──────────────────────────────────────────────────

    def test_forward_raises_without_physics(self) -> None:
        """EDRModel requires the physics tensor — None must raise ValueError."""
        feat = self._make_features()
        with pytest.raises(ValueError, match="physics.*required|required.*physics"):
            self.model(feat, None)

    # ── Phase switching ──────────────────────────────────────────────────────

    def test_phase1_freezes_inertia_coriolis(self) -> None:
        """In phase 1, inertia and Coriolis params should have requires_grad=False."""
        self.model.set_phase(1)
        for name, param in self.model.inertia_net.named_parameters():
            assert not param.requires_grad, (
                f"Phase 1: inertia_net.{name}.requires_grad should be False"
            )
        for name, param in self.model.coriolis_net.named_parameters():
            assert not param.requires_grad, (
                f"Phase 1: coriolis_net.{name}.requires_grad should be False"
            )

    def test_phase1_keeps_gravity_friction_active(self) -> None:
        """In phase 1, gravity and friction params should have requires_grad=True."""
        self.model.set_phase(1)
        for name, param in self.model.gravity_net.named_parameters():
            assert param.requires_grad, (
                f"Phase 1: gravity_net.{name}.requires_grad should be True"
            )
        for name, param in self.model.friction_net.named_parameters():
            assert param.requires_grad, (
                f"Phase 1: friction_net.{name}.requires_grad should be True"
            )

    def test_phase2_all_trainable(self) -> None:
        """In phase 2, all parameters must be trainable."""
        self.model.set_phase(2)
        for name, param in self.model.named_parameters():
            assert param.requires_grad, (
                f"Phase 2: {name}.requires_grad should be True, but is False"
            )

    def test_invalid_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="phase must be 1 or 2"):
            self.model.set_phase(3)

    def test_phase_property(self) -> None:
        self.model.set_phase(1)
        assert self.model.phase == 1
        self.model.set_phase(2)
        assert self.model.phase == 2

    # ── Structural properties preserved end-to-end ───────────────────────────

    def test_inertia_symmetric_through_model(self) -> None:
        """δM symmetry holds even when accessed through the assembled model."""
        q        = _rand(_BATCH, _N_JOINTS)
        delta_M  = self.model.inertia_net.compute_delta_M(q)
        max_asym = (delta_M - delta_M.transpose(1, 2)).abs().max().item()
        assert max_asym < _ATOL_EXACT, (
            f"InertiaCorrection accessed via EDRModel: max |δM - δM^T| = {max_asym:.2e}"
        )

    def test_friction_odd_through_model(self) -> None:
        """Friction odd-function symmetry holds when accessed through the model."""
        qd      = _rand(_BATCH, _N_JOINTS)
        out_pos = self.model.friction_net(qd)
        out_neg = self.model.friction_net(-qd)
        max_viol = (out_pos + out_neg).abs().max().item()
        assert max_viol < _ATOL_EXACT, (
            f"FrictionCorrection accessed via EDRModel: "
            f"max |δτ_f(q̇) + δτ_f(−q̇)| = {max_viol:.2e}"
        )

    def test_coriolis_zero_at_zero_velocity_through_model(self) -> None:
        """Coriolis vanishing at q̇=0 holds through assembled model."""
        q  = _rand(_BATCH, _N_JOINTS)
        qd = torch.zeros(_BATCH, _N_JOINTS)
        # Use the model's public feature-construction API — the same one forward() uses.
        inputs = self.model.build_correction_inputs(q, qd)
        out = self.model.coriolis_net(inputs["coriolis_input"], qd)
        max_abs = out.abs().max().item()
        assert max_abs < 1e-7, (
            f"CoriolisCorrection via EDRModel at q̇=0: max output = {max_abs:.2e}"
        )

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_no_nan_zero_velocity(self) -> None:
        """Zero-velocity inputs must not produce NaN anywhere in the forward pass."""
        feat = _rand(_BATCH, _N_JOINTS * 3)
        # Zero out the q̇ slice (columns 5:10).
        feat[:, 5:10] = 0.0
        phys = self._make_physics()
        out  = self.model(feat, phys)
        assert torch.isfinite(out).all(), (
            "EDRModel forward produced NaN/Inf when q̇ = 0"
        )

    def test_batch_size_1(self) -> None:
        """Single-sample forward pass must work (no batch-norm issues)."""
        feat = self._make_features(B=1)
        phys = self._make_physics(B=1)
        out  = self.model(feat, phys)
        assert out.shape == (1, _N_JOINTS), f"Expected shape (1, {_N_JOINTS}), got {out.shape}"

    def test_wrong_feature_dim_raises(self) -> None:
        feat_bad = _rand(_BATCH, _N_JOINTS * 3 + 1)
        phys     = self._make_physics()
        with pytest.raises(ValueError, match="features last dim"):
            self.model(feat_bad, phys)

    def test_wrong_physics_dim_raises(self) -> None:
        feat     = self._make_features()
        phys_bad = _rand(_BATCH, _N_JOINTS * 4 + 1)
        with pytest.raises(ValueError, match="physics last dim"):
            self.model(feat, phys_bad)

    def test_batch_mismatch_raises(self) -> None:
        feat = self._make_features(B=10)
        phys = self._make_physics(B=11)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            self.model(feat, phys)

    # ── Parameter count ──────────────────────────────────────────────────────

    def test_parameter_count_by_component(self) -> None:
        comp = self.model.count_parameters_by_component()
        assert set(comp.keys()) == {"gravity", "inertia", "coriolis", "friction", "total"}, (
            f"Unexpected keys: {set(comp.keys())}"
        )
        assert comp["total"] > 0, "Total parameter count is zero"

    def test_repr_contains_phase(self) -> None:
        r = repr(self.model)
        assert "phase=" in r, f"EDRModel repr missing 'phase=' field:\n{r}"

    def test_repr_contains_all_nets(self) -> None:
        r = repr(self.model)
        for net_name in ("gravity_net", "inertia_net", "coriolis_net", "friction_net"):
            assert net_name in r, f"EDRModel repr missing '{net_name}':\n{r}"


class TestEDRModelPhysCond:
    """Phase-1 revamp: physics-conditioned residual (use_phys_cond=True).

    The analytic decomposition is threaded into the δ-nets.  Every exact
    structural guarantee must still hold, and the model must still start
    exactly at τ_phys (near-zero init unchanged — only input dims grew).
    """

    def setup_method(self) -> None:
        torch.manual_seed(_SEED)
        # Exercise the harder path: phys-cond *and* trig features together.
        self.model = EDRModel(
            n_joints=_N_JOINTS,
            q_mean=[0.0, 0.1, -0.2, 0.3, -0.1],
            q_std=[1.0, 0.5, 1.2, 0.8, 0.9],
            use_phys_cond=True,
        )
        self.model.eval()

    def _feat(self, B: int = _BATCH) -> torch.Tensor:
        return _rand(B, _N_JOINTS * 3)

    def _phys(self, B: int = _BATCH) -> torch.Tensor:
        return _rand(B, _N_JOINTS * 4)

    def test_phys_cond_starts_at_physics(self) -> None:
        """With phys-conditioning, τ̂ = τ_phys exactly at init (all δ ≈ 0)."""
        feat = self._feat()
        phys = self._phys()
        tau_hat = self.model(feat, phys)
        n = _N_JOINTS
        tau_phys = (
            phys[:, 0:n] + phys[:, n:2*n] + phys[:, 2*n:3*n] + phys[:, 3*n:4*n]
        )
        max_diff = (tau_hat - tau_phys).abs().max().item()
        assert max_diff < _ATOL_INIT, (
            f"phys-cond EDRModel at init: max |τ̂ − τ_phys| = {max_diff:.2e} "
            f"(tolerance {_ATOL_INIT:.0e})"
        )

    def test_phys_cond_output_finite(self) -> None:
        out = self.model(self._feat(), self._phys())
        assert torch.isfinite(out).all(), "phys-cond forward produced NaN/Inf"

    def test_phys_cond_friction_still_odd(self) -> None:
        """Threading |τ_f| keeps δτ_f exactly odd in q̇."""
        qd    = _rand(_BATCH, _N_JOINTS)
        tau_f = _rand(_BATCH, _N_JOINTS)
        out_pos = self.model.friction_net(qd, tau_f=tau_f)
        out_neg = self.model.friction_net(-qd, tau_f=tau_f)
        max_viol = (out_pos + out_neg).abs().max().item()
        assert max_viol < _ATOL_EXACT, (
            f"phys-cond friction not odd: max |δτ_f(q̇)+δτ_f(−q̇)| = {max_viol:.2e}"
        )

    def test_phys_cond_inertia_still_symmetric(self) -> None:
        """δM stays exactly symmetric under phys-conditioned inertia input."""
        q  = _rand(_BATCH, _N_JOINTS)
        qd = _rand(_BATCH, _N_JOINTS)
        inputs  = self.model.build_correction_inputs(q, qd, self._phys())
        delta_M = self.model.inertia_net.compute_delta_M(inputs["inertia_input"])
        max_asym = (delta_M - delta_M.transpose(1, 2)).abs().max().item()
        assert max_asym < _ATOL_EXACT, (
            f"phys-cond δM not symmetric: max |δM − δMᵀ| = {max_asym:.2e}"
        )

    def test_phys_cond_friction_requires_tau_f(self) -> None:
        """use_phys_cond friction must fail loudly if tau_f is omitted."""
        qd = _rand(_BATCH, _N_JOINTS)
        with pytest.raises(ValueError, match="tau_f was not provided"):
            self.model.friction_net(qd)

    def test_phys_cond_param_budget(self) -> None:
        """Phys-conditioning adds only ~input weights — stays well under 50k."""
        total = self.model.count_parameters_by_component()["total"]
        assert total < 50_000, (
            f"phys-cond EDRModel total params {total} exceeds 50k budget."
        )


class TestPerComponentRegularization:
    """Phase 5: capacity-aware per-component correction-reg + decay."""

    def _reg_terms(self):
        dg  = _rand(_BATCH, _N_JOINTS)
        dM  = _rand(_BATCH, _N_JOINTS, _N_JOINTS)
        dC  = _rand(_BATCH, _N_JOINTS)
        df  = _rand(_BATCH, _N_JOINTS)
        return _correction_reg_loss(dg, dM, dC, df, n_joints=_N_JOINTS)

    def test_reg_loss_returns_per_component_dict(self) -> None:
        r = self._reg_terms()
        assert set(r.keys()) == {"g", "M", "C", "f"}, f"keys={set(r.keys())}"
        for k, v in r.items():
            assert torch.is_tensor(v) and v.ndim == 0 and v.item() >= 0.0, (
                f"component {k} must be a non-negative scalar tensor"
            )

    def test_scalar_fallback_back_compat(self) -> None:
        """Absent the dict, the scalar λ applies to all four (exact back-compat)."""
        hp = {"lambda_correction_reg": 0.07}
        lam = _resolve_component_lambdas(hp, epoch=1, total_epochs=100)
        assert lam == {"g": 0.07, "M": 0.07, "C": 0.07, "f": 0.07}, lam

    def test_per_component_dict_overrides(self) -> None:
        hp = {
            "lambda_correction_reg": 0.1,
            "lambda_correction_reg_per_component": {"g": 0.1, "M": 0.4, "C": 0.4, "f": 0.1},
        }
        lam = _resolve_component_lambdas(hp, epoch=1, total_epochs=100)
        assert lam["M"] == 0.4 and lam["C"] == 0.4 and lam["g"] == 0.1, lam

    def test_cosine_decay_monotone_non_increasing(self) -> None:
        hp = {
            "lambda_correction_reg": 1.0,
            "lambda_correction_decay": "cosine",
            "lambda_correction_decay_min_ratio": 0.25,
        }
        vals = [
            _resolve_component_lambdas(hp, epoch=e, total_epochs=100)["M"]
            for e in (1, 25, 50, 75, 100)
        ]
        assert abs(vals[0] - 1.0) < 1e-9, f"epoch1 should be full λ, got {vals[0]}"
        assert abs(vals[-1] - 0.25) < 1e-9, f"last epoch should hit min ratio, got {vals[-1]}"
        assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])), f"not non-increasing: {vals}"


class TestSingleSourceComputeCorrections:
    """B1: forward and compute_corrections are the SAME function (no drift)."""

    def setup_method(self) -> None:
        torch.manual_seed(_SEED)
        self.model = EDRModel(n_joints=_N_JOINTS, use_phys_cond=True)
        self.model.eval()

    def _io(self, B=_BATCH):
        return _rand(B, _N_JOINTS * 3), _rand(B, _N_JOINTS * 4)

    def test_forward_equals_compute_corrections(self) -> None:
        feat, phys = self._io()
        tau_fwd = self.model(feat, phys)
        tau_cc, deltas = self.model.compute_corrections(feat, phys)
        assert torch.equal(tau_fwd, tau_cc), (
            "forward() must return exactly compute_corrections()[0] — any "
            "divergence is the dual-path drift bug B1 eliminates."
        )
        assert set(deltas) == {
            "delta_g", "delta_M", "delta_M_qdd", "delta_C_qd", "delta_tau_f"
        }, f"unexpected deltas keys: {set(deltas)}"
        assert deltas["delta_M"].shape == (_BATCH, _N_JOINTS, _N_JOINTS)

    def test_compute_corrections_starts_at_physics(self) -> None:
        feat, phys = self._io()
        tau_hat, _ = self.model.compute_corrections(feat, phys)
        n = _N_JOINTS
        tau_phys = phys[:, 0:n] + phys[:, n:2*n] + phys[:, 2*n:3*n] + phys[:, 3*n:4*n]
        assert (tau_hat - tau_phys).abs().max().item() < _ATOL_INIT


class TestBuildCorrectionInputGuards:
    """B2: shape / phys-cond guards fail loudly at the source."""

    def test_wrong_qd_dim_raises(self) -> None:
        m = EDRModel(n_joints=_N_JOINTS)
        q  = _rand(8, _N_JOINTS)
        qd = _rand(8, _N_JOINTS - 1)
        with pytest.raises(ValueError, match="q/qd last dim"):
            m.build_correction_inputs(q, qd)

    def test_phys_cond_requires_physics(self) -> None:
        m = EDRModel(n_joints=_N_JOINTS, use_phys_cond=True)
        q  = _rand(8, _N_JOINTS)
        qd = _rand(8, _N_JOINTS)
        with pytest.raises(ValueError, match="use_phys_cond=True but physics=None"):
            m.build_correction_inputs(q, qd, None)

    def test_no_phys_cond_allows_none(self) -> None:
        """Legacy contract preserved when use_phys_cond=False."""
        m = EDRModel(n_joints=_N_JOINTS)  # use_phys_cond defaults False
        q  = _rand(8, _N_JOINTS)
        qd = _rand(8, _N_JOINTS)
        out = m.build_correction_inputs(q, qd, None)
        assert "gravity_input" in out and "inertia_input" in out


class TestInertiaPSD:
    """A2: δM = L Lᵀ is symmetric AND positive-semidefinite, init-safe."""

    def test_psd_and_symmetric(self) -> None:
        torch.manual_seed(_SEED)
        net = InertiaCorrection(n_joints=_N_JOINTS, psd=True)
        # Drive params away from zero so δM is a generic PSD matrix.
        with torch.no_grad():
            for p in net.net[-1].parameters():
                p.add_(torch.randn_like(p) * 0.5)
        q = _rand(_BATCH, _N_JOINTS)
        dM = net.compute_delta_M(q)
        asym = (dM - dM.transpose(1, 2)).abs().max().item()
        assert asym < _ATOL_EXACT, f"δM(psd) not symmetric: {asym:.2e}"
        eig = torch.linalg.eigvalsh(dM)
        assert eig.min().item() > -1e-5, (
            f"δM(psd) must be PSD, got min eigenvalue {eig.min().item():.2e}"
        )

    def test_psd_near_zero_init(self) -> None:
        torch.manual_seed(_SEED)
        net = InertiaCorrection(n_joints=_N_JOINTS, psd=True)
        dM = net.compute_delta_M(_rand(_BATCH, _N_JOINTS))
        assert dM.abs().max().item() < _ATOL_INIT

    def test_model_psd_starts_at_physics(self) -> None:
        torch.manual_seed(_SEED)
        m = EDRModel(n_joints=_N_JOINTS, inertia_psd=True)
        feat = _rand(_BATCH, _N_JOINTS * 3)
        phys = _rand(_BATCH, _N_JOINTS * 4)
        n = _N_JOINTS
        tau_phys = phys[:, 0:n] + phys[:, n:2*n] + phys[:, 2*n:3*n] + phys[:, 3*n:4*n]
        assert (m(feat, phys) - tau_phys).abs().max().item() < _ATOL_INIT


class TestEDRHPValidation:
    """B3: fail-fast HP validation."""

    def test_bad_friction_form(self) -> None:
        with pytest.raises(ValueError, match="friction_form"):
            _validate_edr_hp({"friction_form": "wat"})

    def test_bad_decay(self) -> None:
        with pytest.raises(ValueError, match="lambda_correction_decay"):
            _validate_edr_hp({"lambda_correction_decay": "linear"})

    def test_bad_per_component_keys(self) -> None:
        with pytest.raises(ValueError, match="unknown keys"):
            _validate_edr_hp({"lambda_correction_reg_per_component": {"g": 0.1, "X": 1}})

    def test_bad_joint_weights(self) -> None:
        with pytest.raises(ValueError, match="joint_loss_weights"):
            _validate_edr_hp({"joint_loss_weights": [1.0, -1.0, 1.0, 1.0, 1.0]})

    def test_valid_corrected_p1_passes(self) -> None:
        # The reverted corrected-P1 config must validate cleanly.
        _validate_edr_hp({
            "friction_form": "mlp", "lambda_correction_decay": "none",
            "lambda_correction_reg_per_component": None,
            "joint_loss_weights": None, "use_phys_cond": True,
            "coriolis_matrix_form": False, "inertia_psd": False,
        })


class TestSpectralNormRobustness:
    """Spectral-norm δ-nets preserve every structural guarantee + near-zero init."""

    def setup_method(self) -> None:
        torch.manual_seed(_SEED)
        self.model = EDRModel(
            n_joints=_N_JOINTS, use_phys_cond=True, spectral_norm=True
        ).eval()

    def _io(self, B=_BATCH):
        return _rand(B, _N_JOINTS * 3), _rand(B, _N_JOINTS * 4)

    def test_starts_at_physics(self) -> None:
        feat, phys = self._io()
        tau, _ = self.model.compute_corrections(feat, phys)
        n = _N_JOINTS
        tphys = phys[:, 0:n] + phys[:, n:2*n] + phys[:, 2*n:3*n] + phys[:, 3*n:4*n]
        assert (tau - tphys).abs().max().item() < _ATOL_INIT

    def test_structural_guarantees_preserved(self) -> None:
        """δM symmetric, δC vanishes at q̇=0, δτ_f odd — all still exact."""
        q  = _rand(_BATCH, _N_JOINTS)
        qd = _rand(_BATCH, _N_JOINTS)
        dM = self.model.inertia_net.compute_delta_M(
            self.model.build_correction_inputs(q, qd, _rand(_BATCH, _N_JOINTS*4))["inertia_input"]
        )
        assert (dM - dM.transpose(1, 2)).abs().max().item() < _ATOL_EXACT
        tf = _rand(_BATCH, _N_JOINTS)  # use_phys_cond ⇒ tau_f required
        op = self.model.friction_net(qd, tau_f=tf)
        on = self.model.friction_net(-qd, tau_f=tf)
        assert (op + on).abs().max().item() < _ATOL_EXACT
        z = torch.zeros(_BATCH, _N_JOINTS)
        ci = self.model.build_correction_inputs(q, z, _rand(_BATCH, _N_JOINTS*4))["coriolis_input"]
        assert self.model.coriolis_net(ci, z).abs().max().item() < 1e-6

    def test_ema_decay_validated(self) -> None:
        from edr_strategy import _validate_edr_hp
        _validate_edr_hp({"ema_decay": 0.9})            # ok
        with pytest.raises(ValueError, match="ema_decay"):
            _validate_edr_hp({"ema_decay": 1.0})
        with pytest.raises(ValueError, match="ema_decay"):
            _validate_edr_hp({"ema_decay": -0.1})


class TestStribeckFriction:
    """Phase 4: structured Coulomb+Stribeck+viscous friction correction."""

    def setup_method(self) -> None:
        torch.manual_seed(_SEED)
        self.net = FrictionCorrection(n_joints=_N_JOINTS, friction_form="stribeck")

    def test_invalid_form_raises(self) -> None:
        with pytest.raises(ValueError, match="friction_form must be"):
            FrictionCorrection(n_joints=_N_JOINTS, friction_form="bogus")

    def test_odd_symmetry(self) -> None:
        """δτ_f(−q̇) = −δτ_f(q̇) exactly, for any parameters."""
        # Push params away from zero so all of F_c/F_s/v_s/F_v are active.
        with torch.no_grad():
            for p in self.net.net[-1].parameters():
                p.add_(torch.randn_like(p) * 0.4)
        qd = _rand(_BATCH, _N_JOINTS)
        viol = (self.net(qd) + self.net(-qd)).abs().max().item()
        assert viol < _ATOL_EXACT, f"Stribeck friction not odd: max viol {viol:.2e}"

    def test_zero_at_zero_velocity(self) -> None:
        qd = torch.zeros(_BATCH, _N_JOINTS)
        out = self.net(qd)
        assert out.abs().max().item() < 1e-6, "Stribeck δτ_f must be 0 at q̇=0"

    def test_zero_init(self) -> None:
        qd = _rand(_BATCH, _N_JOINTS)
        out = self.net(qd)
        assert out.abs().max().item() < _ATOL_INIT, (
            f"Stribeck friction should ≈0 at init, max={out.abs().max().item():.2e}"
        )

    def test_finite_and_no_div0(self) -> None:
        """raw_vs≈0 must not blow up (softplus+eps guards the denominator)."""
        qd = _rand(_BATCH, _N_JOINTS) * 5.0
        out = self.net(qd)
        assert torch.isfinite(out).all(), "Stribeck friction produced NaN/Inf"

    def test_non_monotone_shape_with_stribeck_dip(self) -> None:
        """With F_s>0 the |δτ_f| vs |q̇| curve is non-monotone (Stribeck dip)."""
        torch.manual_seed(_SEED)
        net = FrictionCorrection(n_joints=1, friction_form="stribeck")
        # Force a clear Coulomb + Stribeck + small viscous structure on the
        # final layer bias (F_c>0, F_s>0, v_s moderate, F_v small).
        with torch.no_grad():
            fin = net.net[-1]
            fin.weight.zero_()
            fin.bias.copy_(torch.tensor([0.5, 0.8, 0.0, 0.02]))  # F_c,F_s,raw_vs,F_v
        speeds = torch.linspace(0.01, 3.0, 60).unsqueeze(-1)  # (60,1) q̇>0
        mag = net(speeds).abs().squeeze(-1)
        # Coulomb+Stribeck: |δτ_f| starts high (static), dips, then viscous rises.
        assert mag.argmin().item() not in (0, len(mag) - 1), (
            "Expected an interior Stribeck minimum (non-monotone curve)"
        )


class TestJointWeightRebalancing:
    """Per-joint training-loss weights (the per-joint-test-error lever)."""

    def test_none_is_uniform(self) -> None:
        w = _resolve_joint_weights({}, torch.device("cpu"))
        assert torch.allclose(w, torch.ones(_N_JOINTS)), f"None must give uniform, got {w}"

    def test_mean_normalised_to_one(self) -> None:
        hp = {"joint_loss_weights": [1.0, 2.2, 1.4, 0.6, 1.7]}
        w = _resolve_joint_weights(hp, torch.device("cpu"))
        assert abs(w.mean().item() - 1.0) < 1e-6, f"mean must be 1, got {w.mean().item()}"
        # Ratios preserved (hard joint 2 stays the largest).
        assert w.argmax().item() == 1 and w.argmin().item() == 3, w

    def test_positivity_guard(self) -> None:
        with pytest.raises(ValueError, match="must be all > 0"):
            _resolve_joint_weights({"joint_loss_weights": [1.0, 0.0, 1.0, 1.0, 1.0]},
                                   torch.device("cpu"))


class TestEDRModelWithTrigFeatures:
    """Tests for EDRModel with sin/cos augmented gravity input."""

    def setup_method(self) -> None:
        torch.manual_seed(_SEED)
        self.model = EDRModel(
            n_joints=_N_JOINTS,
            gravity_hidden=(32, 32),
            inertia_hidden=(32, 32),
            coriolis_hidden=(32, 32),
            friction_hidden=(16, 16),
            q_mean=[0.0, 0.1, -0.2, 0.3, -0.1],
            q_std=[1.0, 0.5, 1.2, 0.8, 0.9],
        )
        self.model.eval()

    def test_forward_shape(self) -> None:
        feat = _rand(_BATCH, _N_JOINTS * 3)
        phys = _rand(_BATCH, _N_JOINTS * 4)
        out = self.model(feat, phys)
        assert out.shape == (_BATCH, _N_JOINTS), (
            f"EDRModel+trig output shape should be ({_BATCH}, {_N_JOINTS}), got {out.shape}"
        )

    def test_forward_finite(self) -> None:
        feat = _rand(_BATCH, _N_JOINTS * 3)
        phys = _rand(_BATCH, _N_JOINTS * 4)
        out = self.model(feat, phys)
        assert torch.isfinite(out).all(), "EDRModel+trig output contains NaN/Inf"

    def test_starts_at_physics(self) -> None:
        """With trig features, model still starts at τ_phys (all δ-nets ≈ 0)."""
        feat = _rand(_BATCH, _N_JOINTS * 3)
        phys = _rand(_BATCH, _N_JOINTS * 4)
        tau_hat = self.model(feat, phys)
        n = _N_JOINTS
        tau_phys = phys[:, 0:n] + phys[:, n:2*n] + phys[:, 2*n:3*n] + phys[:, 3*n:4*n]
        max_diff = (tau_hat - tau_phys).abs().max().item()
        assert max_diff < _ATOL_INIT, (
            f"EDRModel+trig at init: max |τ̂ − τ_phys| = {max_diff:.2e}  (tol: {_ATOL_INIT:.0e})"
        )

    def test_trig_features_flag(self) -> None:
        """The public use_trig_features property reflects whether norm stats were supplied."""
        assert self.model.use_trig_features is True
        assert self.model.q_normalization is not None
        q_mean, q_std = self.model.q_normalization
        assert q_mean.shape == (_N_JOINTS,) and q_std.shape == (_N_JOINTS,)

        model_no_trig = EDRModel(n_joints=_N_JOINTS)
        assert model_no_trig.use_trig_features is False
        assert model_no_trig.q_normalization is None


# ===========================================================================
# Pure-function tests: adaptive phase-2 plateau detection
# ===========================================================================


class TestShouldTransitionToPhase2:
    """Exhaustively exercise the plateau-detection pure function."""

    _HP_DEFAULT = {
        "phase2_plateau_window":    5,
        "phase2_plateau_threshold": 5e-3,
        "phase2_min_epoch":         3,
        "phase2_max_epoch":         25,
    }

    def test_no_transition_in_phase_2(self) -> None:
        """If we're already in phase 2, never recommend another transition."""
        should, _ = _should_transition_to_phase2([0.1]*10, self._HP_DEFAULT, 30, current_phase=2)
        assert should is False

    def test_manual_override_triggers(self) -> None:
        """When phase2_start_epoch is set (int), it wins over adaptive logic."""
        hp = {"phase2_start_epoch": 7}
        assert _should_transition_to_phase2([], hp, 6, 1) == (False,  "before manual phase2_start_epoch") or \
               _should_transition_to_phase2([], hp, 6, 1)[0] is False
        should, reason = _should_transition_to_phase2([], hp, 7, 1)
        assert should is True
        assert "manual override" in reason

    def test_before_min_epoch_holds(self) -> None:
        """Never trigger before min_epoch regardless of history."""
        hist = [0.10, 0.099, 0.0985]   # clearly plateauing
        should, reason = _should_transition_to_phase2(hist, self._HP_DEFAULT, 2, 1)
        assert should is False
        assert "min_epoch" in reason

    def test_insufficient_history(self) -> None:
        """If we don't yet have window+1 points of history, defer."""
        hist = [0.09, 0.088, 0.087]    # only 3 points, window=5 → need 6
        should, reason = _should_transition_to_phase2(hist, self._HP_DEFAULT, 4, 1)
        assert should is False
        assert "not enough history" in reason

    def test_still_improving_no_trigger(self) -> None:
        """If val_rmse is still dropping >=0.5% over window, don't transition."""
        # 10% improvement over 5 epochs — well above 0.5% threshold.
        hist = [0.100, 0.098, 0.096, 0.094, 0.092, 0.090]
        should, reason = _should_transition_to_phase2(hist, self._HP_DEFAULT, 7, 1)
        assert should is False
        assert "still improving" in reason

    def test_plateau_triggers(self) -> None:
        """Flat val_rmse over window → transition triggers."""
        hist = [0.090, 0.0900, 0.0900, 0.0900, 0.0900, 0.0900]
        should, reason = _should_transition_to_phase2(hist, self._HP_DEFAULT, 7, 1)
        assert should is True
        assert "plateau" in reason

    def test_slight_improvement_below_threshold_triggers(self) -> None:
        """Improvement below 0.5% counts as plateau."""
        hist = [0.0900, 0.0899, 0.0898, 0.0898, 0.0898, 0.08982]  # ~0.02%
        should, reason = _should_transition_to_phase2(hist, self._HP_DEFAULT, 7, 1)
        assert should is True

    def test_safety_fallback_fires_at_max_epoch(self) -> None:
        """At phase2_max_epoch, always force transition even if still improving."""
        # Still rapidly improving, but at epoch == max_epoch.
        hist = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05]
        should, reason = _should_transition_to_phase2(hist, self._HP_DEFAULT, 25, 1)
        assert should is True
        assert "safety fallback" in reason

    def test_non_positive_reference_guards(self) -> None:
        """Degenerate val_rmse=0 in history must not divide by zero."""
        hist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        should, reason = _should_transition_to_phase2(hist, self._HP_DEFAULT, 7, 1)
        assert should is False
        assert "non-positive" in reason


# ===========================================================================
# Friction smooth-abs gradient test (T3.2)
# ===========================================================================


class TestFrictionSmoothGradient:
    """Verify the friction network is differentiable at qd = 0 (smooth |qd|)."""

    def test_gradient_finite_at_zero(self) -> None:
        torch.manual_seed(_SEED)
        net = FrictionCorrection(n_joints=_N_JOINTS, hidden_sizes=(16, 16))
        qd = torch.zeros(1, _N_JOINTS, requires_grad=True)
        out = net(qd)
        loss = out.sum()
        loss.backward()
        assert qd.grad is not None
        assert torch.isfinite(qd.grad).all(), (
            "Friction gradient at q̇=0 is non-finite — smooth |q̇| approximation "
            "is broken."
        )
        # Since δτ_f(0) = 0 · h(|0|) = 0 and δτ_f = q̇ · h(|q̇|), the gradient
        # at q̇=0 equals h(√ε) — small but finite, not NaN/Inf.

    def test_odd_symmetry_with_smooth_abs(self) -> None:
        """Smooth |q̇| must preserve exact odd symmetry of δτ_f."""
        torch.manual_seed(_SEED)
        net = FrictionCorrection(n_joints=_N_JOINTS, hidden_sizes=(16, 16))
        qd = _rand(_BATCH, _N_JOINTS)
        out_pos = net(qd)
        out_neg = net(-qd)
        max_violation = (out_pos + out_neg).abs().max().item()
        assert max_violation < _ATOL_EXACT, (
            f"Smooth-abs friction violates odd symmetry: max |f(qd)+f(-qd)| = {max_violation:.2e}"
        )
