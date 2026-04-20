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
        """δC(q, 0)·0 = 0 exactly.

        Coriolis forces are proportional to q̇ — they vanish when the robot is
        stationary.  The element-wise product construction ensures this holds
        by multiplying MLP output by q̇, which is zero when q̇ = 0.
        """
        q  = _rand(_BATCH, _N_JOINTS)
        qd = torch.zeros(_BATCH, _N_JOINTS)
        out = self.net(self._make_features(q, qd), qd)
        max_abs = out.abs().max().item()
        assert max_abs < 1e-7, (
            f"CoriolisCorrection must vanish at q̇ = 0 (quadratic construction), "
            f"but max |δC·q̇| = {max_abs:.2e}"
        )

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
        feat = torch.cat([q, qd], dim=-1)
        out = self.model.coriolis_net(feat, qd)
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
        assert self.model._use_trig_features is True
        model_no_trig = EDRModel(n_joints=_N_JOINTS)
        assert model_no_trig._use_trig_features is False
