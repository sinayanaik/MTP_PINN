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
    _correction_gain,
    _correction_reg_loss,
    _resolve_component_lambdas,
    _resolve_joint_weights,
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

    # ── Smooth capacity gate γ (R3) ──────────────────────────────────────────

    def test_all_params_trainable_from_start(self) -> None:
        """No phase freeze: every parameter trains from epoch 1 (R3)."""
        for name, param in self.model.named_parameters():
            assert param.requires_grad, (
                f"{name}.requires_grad should be True (γ-gate, no freeze)"
            )

    def test_gamma_zero_recovers_nominal_physics(self) -> None:
        """γ=0 ⇒ inertia+Coriolis contribution gated off ⇒ τ̂ ≈ τ_phys + δg + δτ_f.

        At init δg, δτ_f ≈ 0 too, so τ̂ = τ_phys exactly even with γ=0 — the
        ramp can safely start at 0 with no discontinuity.
        """
        self.model.set_correction_gain(0.0)
        feat, phys = self._make_features(), self._make_physics()
        n = _N_JOINTS
        tau_phys = phys[:, 0:n] + phys[:, n:2*n] + phys[:, 2*n:3*n] + phys[:, 3*n:4*n]
        max_diff = (self.model(feat, phys) - tau_phys).abs().max().item()
        assert max_diff < _ATOL_INIT, (
            f"γ=0 at init should give τ̂=τ_phys, max diff {max_diff:.2e}"
        )

    def test_gamma_scales_inertia_coriolis_contribution(self) -> None:
        """The (δM·q̈+δC·q̇) contribution scales linearly with γ."""
        # Give the inertia net a non-trivial output so the gate is observable.
        with torch.no_grad():
            for p in self.model.inertia_net.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        feat, phys = self._make_features(), self._make_physics()
        self.model.set_correction_gain(0.0)
        _, d0 = self.model.compute_corrections(feat, phys)
        base = self.model(feat, phys)
        self.model.set_correction_gain(1.0)
        full = self.model(feat, phys)
        self.model.set_correction_gain(0.5)
        half = self.model(feat, phys)
        # half should sit ~midway between gated-off and full.
        mid = 0.5 * (base + full)
        assert torch.allclose(half, mid, atol=1e-5), "γ does not scale linearly"

    def test_invalid_gamma_raises(self) -> None:
        with pytest.raises(ValueError, match=r"γ must be"):
            self.model.set_correction_gain(1.5)

    # ── Structural robustness: SPD inertia + Christoffel passivity ───────────

    def test_corrected_inertia_is_spd(self) -> None:
        """R2: δM = LLᵀ is PSD ⇒ M̃ = M_nom + δM is SPD (M_nom SPD by physics).

        We verify δM itself is symmetric PSD (eigenvalues ≥ 0); adding the
        SPD nominal M then keeps M̃ SPD for every parameter value.
        """
        assert self.model._inertia_psd, "default EDRModel should be inertia_psd=True"
        with torch.no_grad():
            for p in self.model.inertia_net.parameters():
                p.add_(torch.randn_like(p) * 0.3)   # arbitrary (trained-like) weights
        q = _rand(_BATCH, _N_JOINTS)
        dM = self.model.inertia_net.compute_delta_M(q)
        asym = (dM - dM.transpose(1, 2)).abs().max().item()
        assert asym < 1e-5, f"δM not symmetric: max|δM-δMᵀ|={asym:.2e}"
        eigmin = torch.linalg.eigvalsh(dM).min().item()
        assert eigmin > -1e-5, f"δM not PSD: min eigenvalue {eigmin:.2e}"

    def test_structural_coriolis_is_quadratic_in_qd(self) -> None:
        """δC·q̇ is the Christoffel contraction Σ_jk c_ijk q̇_j q̇_k, hence
        exactly *quadratic* in q̇: scaling q̇→αq̇ scales δC·q̇ by α²
        (centripetal/Coriolis homogeneity — a physical invariant the old
        independent δC network did not satisfy)."""
        assert self.model._coriolis_structural
        with torch.no_grad():
            for p in self.model.inertia_net.parameters():
                p.add_(torch.randn_like(p) * 0.2)
        q  = _rand(_BATCH, _N_JOINTS)
        qd = _rand(_BATCH, _N_JOINTS)
        base = self.model._structural_delta_C_qd(q, qd, None)
        for alpha in (0.5, 2.0, -1.0):
            scaled = self.model._structural_delta_C_qd(q, alpha * qd, None)
            assert torch.allclose(scaled, (alpha ** 2) * base, atol=1e-5), (
                f"δC·q̇ not quadratic in q̇ at α={alpha}"
            )

    def test_structural_coriolis_zero_for_constant_inertia(self) -> None:
        """If δM does not depend on q (zero-init final layer ⇒ δM≡0, ∂δM/∂q=0)
        the Christoffel correction is exactly zero — no spurious Coriolis."""
        q  = _rand(_BATCH, _N_JOINTS)
        qd = _rand(_BATCH, _N_JOINTS)
        out = self.model._structural_delta_C_qd(q, qd, None)   # fresh model: δM≈0
        # δM final layer is near-zero init (std=1e-4), not exactly 0, so ∂δM/∂q
        # is ~1e-7 — negligible vs the O(1) physics baseline.
        assert out.abs().max().item() < 1e-5, (
            f"Christoffel δC should ≈vanish for q-independent δM, got {out.abs().max():.2e}"
        )

    def test_skew_symmetry_Mdot_minus_2C(self) -> None:
        """Passivity: q̇ᵀ (δṀ − 2 δC) q̇ = 0 for the corrected terms.

        δṀ = Σ_k (∂δM/∂q_k) q̇_k.  With δC the Christoffel partner of δM the
        quadratic form vanishes identically (energy-conserving correction).
        """
        from torch.func import jacrev, vmap
        with torch.no_grad():
            for p in self.model.inertia_net.parameters():
                p.add_(torch.randn_like(p) * 0.25)
        q  = _rand(_BATCH, _N_JOINTS)
        qd = _rand(_BATCH, _N_JOINTS)

        def _dM(qr):
            return self.model.inertia_net.compute_delta_M(qr.unsqueeze(0)).squeeze(0)
        J = vmap(jacrev(_dM))(q)                              # ∂δM_ij/∂q_k
        dM_dt = torch.einsum("bijk,bk->bij", J, qd)           # δṀ
        # δC matrix from Christoffel: C_ij = Σ_k c_ijk q̇_k
        c = 0.5 * (J + J.transpose(2, 3) - J.permute(0, 2, 3, 1))
        Cmat = torch.einsum("bijk,bk->bij", c, qd)
        S = dM_dt - 2.0 * Cmat
        quad = torch.einsum("bi,bij,bj->b", qd, S, qd)        # q̇ᵀ S q̇
        assert quad.abs().max().item() < 1e-4, (
            f"passivity violated: max |q̇ᵀ(Ṁ̇−2C)q̇| = {quad.abs().max().item():.2e}"
        )

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
        """δC·q̇ vanishes at q̇=0 (structural Christoffel form: quadratic in q̇)."""
        q  = _rand(_BATCH, _N_JOINTS)
        qd = torch.zeros(_BATCH, _N_JOINTS)
        out = self.model._structural_delta_C_qd(q, qd, None)
        max_abs = out.abs().max().item()
        assert max_abs < 1e-7, (
            f"structural δC·q̇ at q̇=0: max output = {max_abs:.2e}"
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

    def test_repr_contains_gamma(self) -> None:
        r = repr(self.model)
        assert "γ=" in r, f"EDRModel repr missing 'γ=' field:\n{r}"

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
        phys = _rand(_BATCH, _N_JOINTS * 4)
        assert self.model._structural_delta_C_qd(q, z, phys).abs().max().item() < 1e-6

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
# Pure-function tests: smooth capacity-gate ramp γ(epoch) (R3)
# ===========================================================================


class TestCorrectionGainRamp:
    """The γ ramp replaces the two-phase plateau detector — must be a smooth,
    monotone 0→1 cosine that then stays saturated (no discontinuity)."""

    def test_starts_at_zero(self) -> None:
        assert _correction_gain(1, 1000, 0.30) == 0.0

    def test_saturates_at_one(self) -> None:
        # ramp ends at ceil(0.30*1000)=300 → γ=1 from epoch 301 onward.
        assert _correction_gain(301, 1000, 0.30) == 1.0
        assert _correction_gain(1000, 1000, 0.30) == 1.0

    def test_monotone_non_decreasing(self) -> None:
        vals = [_correction_gain(e, 1000, 0.30) for e in range(1, 1001)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), (
            "γ ramp must be monotone non-decreasing"
        )
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_midpoint_is_half(self) -> None:
        # Half-cosine: at progress=0.5 (epoch ≈ 1 + 0.5*ramp) γ = 0.5.
        ramp = 300  # ceil(0.30*1000)
        mid = _correction_gain(1 + ramp // 2, 1000, 0.30)
        assert abs(mid - 0.5) < 0.02, f"γ at ramp midpoint should be ~0.5, got {mid}"

    def test_no_curriculum_when_frac_zero(self) -> None:
        """ramp_frac<=0 ⇒ γ≡1 (full capacity from epoch 1)."""
        assert _correction_gain(1, 1000, 0.0) == 1.0
        assert _correction_gain(1, 1000, -0.5) == 1.0

    def test_continuous_no_jump(self) -> None:
        """Successive-epoch γ deltas are tiny — no phase-style discontinuity."""
        deltas = [
            _correction_gain(e + 1, 1000, 0.30) - _correction_gain(e, 1000, 0.30)
            for e in range(1, 1000)
        ]
        assert max(deltas) < 0.02, (
            f"γ ramp must be smooth; largest single-epoch jump {max(deltas):.4f}"
        )


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
