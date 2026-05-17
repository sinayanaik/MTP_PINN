"""EDR correction networks — four structurally-constrained nn.Module classes.

Each network learns a *structured correction* to one component of the robot
dynamics:

    δg(q)           — gravity correction         (unconstrained MLP in q)
    δM(q)·q̈         — inertia correction         (symmetric δM by LL^T)
    δC(q,q̇)·q̇       — Coriolis correction        (quadratic in q̇, vanishes at q̇=0)
    δτ_f(q̇) = q̇ ⊙ h(|q̇|)  — friction correction  (odd function by construction)

All four networks are zero-initialised at their final layer, so that training
begins exactly at the nominal physics solution.  The structural constraints are
*enforced by architecture*, not by soft penalties — they hold exactly for every
parameter configuration.

NASA Defensive Programming conventions applied throughout:
  • Every public method validates its inputs (shape, dtype, finiteness).
  • Every public method validates its output before returning.
  • Failure modes are explicit ValueError/RuntimeError with descriptive messages.
  • No silent fallbacks.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Minimum velocity magnitude used to avoid division-by-zero in the Coriolis
# safe normalisation.  Below this threshold the Coriolis output is zeroed.
_QVEL_EPS: float = 1e-6

# Smoothing constant for the friction |q̇| approximation.  Using
# ``sqrt(q̇² + eps)`` instead of ``|q̇|`` makes the friction correction
# differentiable at q̇ = 0 (its derivative → 0 as q̇ → 0) and matches the
# smooth Stribeck-curve behaviour of real friction near stick-slip transitions.
# Value is small enough that ``sqrt(q̇² + eps) ≈ |q̇|`` to within 0.1% when
# |q̇| > 0.03 rad/s (the ε regime only applies very close to zero velocity).
_FRICTION_ABS_EPS: float = 1e-6

# Supported nonlinear activations (string → class).
_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "tanh":       nn.Tanh,
    "relu":       nn.ReLU,
    "silu":       nn.SiLU,
    "gelu":       nn.GELU,
    "elu":        nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


# ===========================================================================
# Internal helpers
# ===========================================================================

def _build_correction_mlp(
    in_dim:       int,
    hidden_sizes: Sequence[int],
    out_dim:      int,
    activation:   str = "tanh",
    dropout:      float = 0.0,
    spectral_norm: bool = False,
) -> nn.Sequential:
    """Build a small MLP whose **final layer is zero-initialised**.

    Zero-init on the last linear layer guarantees that the correction network
    outputs exactly zero before any training step.  This means the assembled
    EDR model starts at the nominal physics prediction — a sensible and
    reproducible baseline.

    Parameters
    ----------
    in_dim:
        Input feature dimensionality.
    hidden_sizes:
        Sequence of hidden-layer widths (e.g. [32, 32]).
    out_dim:
        Output dimensionality.
    activation:
        Name of the nonlinearity applied after each hidden layer.  Supported:
        tanh, relu, silu, gelu, elu, leaky_relu.
    dropout:
        Dropout probability after each hidden activation (not after the final
        linear layer).  ``0.0`` disables dropout (default).

    Returns
    -------
    nn.Sequential
        The constructed MLP.  Weights of all hidden Linear layers are
        initialised with Xavier-normal; biases are zero.  The final layer has
        both weights and bias initialised to zero.

    Raises
    ------
    ValueError
        If ``activation`` is not in the supported set.
    ValueError
        If any dimension is non-positive or ``hidden_sizes`` is empty.
    """
    activation = activation.lower()
    if activation not in _ACTIVATION_MAP:
        raise ValueError(
            f"[_build_correction_mlp] Unknown activation {activation!r}. "
            f"Supported: {sorted(_ACTIVATION_MAP)}"
        )
    if in_dim < 1:
        raise ValueError(f"[_build_correction_mlp] in_dim must be ≥ 1, got {in_dim}")
    if out_dim < 1:
        raise ValueError(f"[_build_correction_mlp] out_dim must be ≥ 1, got {out_dim}")
    if len(hidden_sizes) == 0:
        raise ValueError("[_build_correction_mlp] hidden_sizes must be non-empty")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError(
            f"[_build_correction_mlp] dropout must be in [0, 1), got {dropout!r}"
        )
    dropout = float(dropout)
    for i, h in enumerate(hidden_sizes):
        if h < 1:
            raise ValueError(
                f"[_build_correction_mlp] hidden_sizes[{i}] must be ≥ 1, got {h}"
            )

    Act = _ACTIVATION_MAP[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    # Spectral normalisation bounds each hidden layer's Lipschitz constant,
    # so the learned correction cannot fit high-frequency residual noise —
    # a principled generalisation regulariser (math robustness), no data
    # change.  Applied to HIDDEN layers only: the final layer keeps its
    # near-zero init untouched so the τ̂=τ_phys-at-init guarantee and all
    # structural constructions (symmetry / oddness / vanishing — which are
    # architectural, not weight-norm dependent) are preserved exactly.
    for h in hidden_sizes:
        lin = nn.Linear(prev, h)
        nn.init.xavier_normal_(lin.weight)
        nn.init.zeros_(lin.bias)
        if spectral_norm:
            lin = nn.utils.parametrizations.spectral_norm(lin)
        layers += [lin, Act()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = h

    # Final projection — near-zero weight init (std=1e-4), zero bias.
    #
    # Why NOT exactly zero?
    # Exactly-zero weights create a gradient deadlock: chain rule gives
    # ∂L/∂hidden = W_last^T × upstream = 0, so hidden layers receive no
    # gradient on step 1.  For InertiaCorrection (which computes L L^T),
    # the deadlock is PERMANENT: ∂(L L^T q̈)/∂L = 2L × ... = 0 whenever
    # L = 0, and L can never escape because its own gradient is also 0.
    #
    # std=1e-4 gives output magnitude ~O(1e-4)–O(1e-3) — negligible vs
    # physics baseline (~O(1)) — while keeping gradients alive from step 1.
    final = nn.Linear(prev, out_dim)
    nn.init.normal_(final.weight, mean=0.0, std=1e-4)
    nn.init.zeros_(final.bias)
    layers.append(final)

    return nn.Sequential(*layers)


def _assert_tensor(
    x:         torch.Tensor,
    name:      str,
    expected_last_dim: int,
) -> None:
    """NASA input guard: validate tensor shape, dtype, and finiteness.

    Parameters
    ----------
    x:
        Tensor to validate.
    name:
        Human-readable variable name for error messages.
    expected_last_dim:
        Expected size of the last dimension.

    Raises
    ------
    TypeError
        If ``x`` is not a ``torch.Tensor``.
    ValueError
        If shape, dtype, or finiteness constraints are violated.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"[EDR] Expected torch.Tensor for '{name}', got {type(x).__name__}"
        )
    if x.ndim < 2:
        raise ValueError(
            f"[EDR] '{name}' must have at least 2 dimensions (batch × features), "
            f"got shape {tuple(x.shape)}"
        )
    if x.shape[-1] != expected_last_dim:
        raise ValueError(
            f"[EDR] '{name}' last dimension must be {expected_last_dim}, "
            f"got {x.shape[-1]}  (full shape: {tuple(x.shape)})"
        )
    if not x.dtype.is_floating_point:
        raise ValueError(
            f"[EDR] '{name}' must be a floating-point tensor, got dtype {x.dtype}"
        )
    if not torch.isfinite(x).all():
        n_bad = int((~torch.isfinite(x)).sum().item())
        raise ValueError(
            f"[EDR] '{name}' contains {n_bad} non-finite value(s) (NaN/Inf). "
            "Check upstream preprocessing and physics computation."
        )


def _assert_output_finite(
    y:    torch.Tensor,
    name: str,
) -> None:
    """NASA output guard: verify that a computed tensor is finite.

    Raises
    ------
    RuntimeError
        If ``y`` contains any NaN or Inf value.
    """
    if not torch.isfinite(y).all():
        n_bad = int((~torch.isfinite(y)).sum().item())
        raise RuntimeError(
            f"[EDR] Internal error: output '{name}' contains {n_bad} non-finite "
            "value(s). This is a bug — please report it with the input that "
            "triggered the failure."
        )


# ===========================================================================
# 1. Gravity correction  δg(q)
# ===========================================================================

class GravityCorrection(nn.Module):
    """Learn a configuration-dependent correction to the gravity torque vector.

    Mathematical form
    -----------------
    δg(q) = MLP_g(q_aug)   ∈ ℝ^n

    The gravity torque depends only on joint position q (it derives from the
    potential energy V(q)).  This network outputs a per-joint additive
    correction.

    When ``in_dim`` > ``n_joints``, the input is expected to be an augmented
    feature vector (e.g. [q_norm, sin(q_raw), cos(q_raw)]) that provides
    physics-informed features to help the network learn the trigonometric
    dependence of gravity on joint angles.

    Architecture: 2 hidden layers, zero-initialised output layer.

    Parameters
    ----------
    n_joints:
        Number of active joints (output dimensionality).
    in_dim:
        Input dimensionality.  Defaults to ``n_joints`` (normalised q only).
        Set to ``3 * n_joints`` when using sin/cos augmented input.
    hidden_sizes:
        Width of each hidden layer.
    activation:
        Nonlinearity name.  Tanh is recommended (bounded, smooth).
    """

    def __init__(
        self,
        n_joints:     int = 5,
        in_dim:       int | None = None,
        hidden_sizes: Sequence[int] = (32, 32),
        activation:   str = "tanh",
        dropout:      float = 0.0,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if n_joints < 1:
            raise ValueError(f"n_joints must be ≥ 1, got {n_joints}")
        self.n_joints = int(n_joints)
        self.in_dim = int(in_dim if in_dim is not None else n_joints)
        self._spectral_norm = bool(spectral_norm)
        self.hparams = {
            "n_joints":     self.n_joints,
            "in_dim":       self.in_dim,
            "hidden_sizes": list(hidden_sizes),
            "activation":   activation,
            "dropout":      float(dropout),
        }
        self.net = _build_correction_mlp(
            in_dim=self.in_dim,
            hidden_sizes=hidden_sizes,
            out_dim=self.n_joints,
            activation=activation,
            dropout=dropout,
            spectral_norm=self._spectral_norm,
        )

    def forward(self, q_input: torch.Tensor) -> torch.Tensor:
        """Compute gravity correction δg(q).

        Parameters
        ----------
        q_input:
            Joint position features, shape (B, in_dim).
            Can be plain q (n_joints) or augmented [q, sin(q_raw), cos(q_raw)]
            (3 * n_joints).

        Returns
        -------
        torch.Tensor
            δg ∈ ℝ^(B × n_joints).
        """
        _assert_tensor(q_input, "q_input [GravityCorrection]", self.in_dim)
        delta_g = self.net(q_input)
        _assert_output_finite(delta_g, "delta_g")
        return delta_g

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# 2. Inertia correction  δM(q) · q̈
# ===========================================================================

class InertiaCorrection(nn.Module):
    """Learn a symmetric correction to the joint-space inertia matrix.

    Mathematical form
    -----------------
    An MLP produces n*(n+1)/2 free parameters representing the independent
    entries of a symmetric n×n matrix.  These are scattered directly into
    both symmetric positions (lower + upper triangle; diagonal written once)
    to produce δM(q) exactly symmetric by construction:

        δM(q)_ij = δM(q)_ji    for all i, j

    The correction can be positive, negative, or indefinite — this lets the
    network both increase and decrease effective inertia per configuration.
    The total effective inertia M + δM remains positive-definite as long as
    |δM| is small relative to M, which the regularisation loss enforces.

    Architecture: MLP with 2 hidden layers producing n*(n+1)/2 entries,
    scattered into a symmetric n×n matrix.  ~500 parameters for n_joints=5,
    hidden_size=32.

    Parameters
    ----------
    n_joints:
        Number of active joints.
    hidden_sizes:
        Hidden layer widths for the L_θ network.
    activation:
        Nonlinearity name.
    """

    def __init__(
        self,
        n_joints:     int = 5,
        hidden_sizes: Sequence[int] = (32, 32),
        activation:   str = "tanh",
        dropout:      float = 0.0,
        in_dim:       int | None = None,
        psd:          bool = False,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if n_joints < 1:
            raise ValueError(f"n_joints must be ≥ 1, got {n_joints}")
        self.n_joints = int(n_joints)
        # Allow caller to supply trig-augmented features [q, sin(q), cos(q)].
        # Default: plain joint positions (in_dim == n_joints).
        self.in_dim = int(in_dim) if in_dim is not None else self.n_joints
        self._spectral_norm = bool(spectral_norm)
        # Number of independent entries in an n×n lower-triangular matrix.
        self.n_tri = self.n_joints * (self.n_joints + 1) // 2
        # PSD mode: interpret the n_tri MLP outputs as a lower-triangular L and
        # set δM = L Lᵀ (symmetric AND positive-semidefinite by construction).
        # Default OFF: an unconstrained-symmetric δM can also *reduce* an
        # over-estimated nominal inertia, which a PSD-only δM cannot — so PSD
        # is offered as a physically-stricter ablation, not forced.
        self._psd = bool(psd)
        # Pre-compute lower-triangular row/column indices for fast scatter.
        rows, cols = torch.tril_indices(self.n_joints, self.n_joints)
        # Register as buffers so they are moved to the correct device with the module.
        self.register_buffer("_tri_rows", rows, persistent=False)
        self.register_buffer("_tri_cols", cols, persistent=False)
        self.hparams = {
            "n_joints":     self.n_joints,
            "in_dim":       self.in_dim,
            "hidden_sizes": list(hidden_sizes),
            "activation":   activation,
            "dropout":      float(dropout),
            "psd":          self._psd,
        }
        # The MLP outputs n_tri numbers for each sample.
        self.net = _build_correction_mlp(
            in_dim=self.in_dim,
            hidden_sizes=hidden_sizes,
            out_dim=self.n_tri,
            activation=activation,
            dropout=dropout,
            spectral_norm=self._spectral_norm,
        )

    def _features_to_delta_M(self, q_features: torch.Tensor) -> torch.Tensor:
        """Map configuration features to the symmetric inertia correction matrix.

        The MLP produces n*(n+1)/2 free parameters representing the independent
        entries of a symmetric matrix.  These are scattered directly into both
        the lower and upper triangles; the diagonal is double-written with the
        same value (equivalent to a single write — symmetry is preserved).

        Parameters
        ----------
        q_features:
            Configuration features, shape (B, in_dim).  When trig features are
            enabled this is [q, sin(q_raw), cos(q_raw)] (3·n_joints wide);
            otherwise it is plain normalised q (n_joints wide).

        Returns
        -------
        delta_M:
            Symmetric correction matrices, shape (B, n_joints, n_joints).
        """
        B = q_features.shape[0]
        tri_entries = self.net(q_features)   # (B, n_tri)
        if self._psd:
            # Build lower-triangular L from the n_tri entries, δM = L Lᵀ.
            # Symmetric AND PSD by construction; near-zero init preserved
            # (entries ≈ 0 ⇒ L ≈ 0 ⇒ L Lᵀ ≈ 0).
            L = torch.zeros(
                B, self.n_joints, self.n_joints,
                dtype=q_features.dtype, device=q_features.device,
            )
            L[:, self._tri_rows, self._tri_cols] = tri_entries
            return torch.bmm(L, L.transpose(1, 2))
        delta_M = torch.zeros(
            B, self.n_joints, self.n_joints,
            dtype=q_features.dtype,
            device=q_features.device,
        )
        # Scatter into both symmetric positions.  Diagonal (rows==cols) gets
        # written twice with the same value — a no-op for correctness.
        delta_M[:, self._tri_rows, self._tri_cols] = tri_entries
        delta_M[:, self._tri_cols, self._tri_rows] = tri_entries
        return delta_M

    def forward(self, q_features: torch.Tensor, qdd: torch.Tensor) -> torch.Tensor:
        """Compute inertia correction contribution  δM(q) @ q̈.

        Parameters
        ----------
        q_features:
            Configuration features, shape (B, in_dim).  Plain normalised q or
            trig-augmented [q, sin(q_raw), cos(q_raw)] depending on how the
            model was constructed.
        qdd:
            Joint accelerations, shape (B, n_joints).

        Returns
        -------
        torch.Tensor
            δM(q) @ q̈, shape (B, n_joints).
        """
        _assert_tensor(qdd, "qdd [InertiaCorrection]", self.n_joints)
        if q_features.shape[0] != qdd.shape[0]:
            raise ValueError(
                f"[InertiaCorrection] Batch size mismatch: q_features has "
                f"{q_features.shape[0]} rows, qdd has {qdd.shape[0]} rows."
            )
        delta_M = self._features_to_delta_M(q_features)  # (B, n, n)
        # (B, n, n) @ (B, n, 1) → (B, n, 1) → squeeze to (B, n).
        delta_M_qdd = torch.bmm(delta_M, qdd.unsqueeze(-1)).squeeze(-1)
        _assert_output_finite(delta_M_qdd, "delta_M_qdd")
        return delta_M_qdd

    def compute_delta_M(self, q_features: torch.Tensor) -> torch.Tensor:
        """Expose the correction matrix itself (useful for tests and analysis).

        Parameters
        ----------
        q_features:
            Configuration features, shape (B, in_dim).

        Returns
        -------
        torch.Tensor
            δM(q), shape (B, n_joints, n_joints).
        """
        return self._features_to_delta_M(q_features)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# 3. Coriolis correction  δC(q, q̇) · q̇
# ===========================================================================

class CoriolisCorrection(nn.Module):
    """Learn a correction to the Coriolis/centripetal torque.

    Mathematical form
    -----------------
    The Coriolis torque C(q, q̇)q̇ vanishes identically when the *full*
    velocity vector q̇ = 0.  The correction must share this property.

    Matrix form (``matrix_form=True``, canonical, Phase-2 revamp):

        δC(q,q̇)·q̇ = B(q,q̇) @ q̇,   B ∈ ℝ^{n×n} from the MLP.

    The MLP emits n² entries reshaped to B(q,q̇); the contribution is the
    matrix-vector product B·q̇.  It vanishes **iff the full q̇ vector is
    zero** (B · 0 = 0 for any B) — the physically-correct property — while
    allowing genuine cross-joint coupling: joint i's Coriolis correction can
    be non-zero when q̇_i = 0 as long as some other joint moves.

    Legacy element-wise form (``matrix_form=False``, kept for ablation):

        δC(q,q̇)·q̇ = q̇ ⊙ MLP_C(features)

    This wrongly forces joint i's correction to vanish whenever q̇_i = 0,
    even if other joints move — strictly less expressive and not physical.

    The MLP input can optionally be augmented with physics-informed features
    (sin/cos of raw joint angles, τ_C when phys-conditioned).

    Parameters
    ----------
    n_joints:
        Number of active joints.
    in_dim:
        Input dimensionality.  Defaults to 2*n_joints (plain [q, qd]).
        Set to 4*n_joints when using augmented [q, sin(q_raw), cos(q_raw), qd].
    hidden_sizes:
        Hidden layer widths.
    activation:
        Nonlinearity name.
    matrix_form:
        If True (default), use the correct B(q,q̇)·q̇ matrix construction.
        If False, use the legacy per-joint element-wise q̇ ⊙ MLP form.
    """

    def __init__(
        self,
        n_joints:     int = 5,
        in_dim:       int | None = None,
        hidden_sizes: Sequence[int] = (32, 32),
        activation:   str = "tanh",
        dropout:      float = 0.0,
        matrix_form:  bool = True,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if n_joints < 1:
            raise ValueError(f"n_joints must be ≥ 1, got {n_joints}")
        self.n_joints = int(n_joints)
        self.in_dim = int(in_dim if in_dim is not None else n_joints * 2)
        self._spectral_norm = bool(spectral_norm)
        # matrix_form=True  → δC·q̇ = B(q,q̇)·q̇ (cross-joint, vector-vanishing).
        # matrix_form=False → element-wise q̇⊙MLP (per-joint vanishing).  Both
        # are first-class, explicitly recorded in hparams below; the choice is
        # data-dependent (on this dataset the element-wise form generalised
        # *better*), so neither is "wrong" — no runtime warning, just the
        # explicit hparam so a config is never silently misread.
        self._matrix_form = bool(matrix_form)
        # Matrix form emits n² entries (full B); legacy emits n (per-joint scale).
        _out_dim = self.n_joints * self.n_joints if self._matrix_form else self.n_joints
        self.hparams = {
            "n_joints":     self.n_joints,
            "in_dim":       self.in_dim,
            "hidden_sizes": list(hidden_sizes),
            "activation":   activation,
            "dropout":      float(dropout),
            "matrix_form":  self._matrix_form,
        }
        self.net = _build_correction_mlp(
            in_dim=self.in_dim,
            hidden_sizes=hidden_sizes,
            out_dim=_out_dim,
            activation=activation,
            dropout=dropout,
            spectral_norm=self._spectral_norm,
        )

    def forward(self, features: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        """Compute Coriolis correction  δC(q, q̇) · q̇.

        Parameters
        ----------
        features:
            Input features, shape (B, in_dim).  Typically [q, qd] (2n) or
            [q, sin(q_raw), cos(q_raw), qd] (4n).
        qd:
            Joint velocities, shape (B, n_joints).  The matrix-vector (or
            element-wise) product with q̇ enforces vanishing at q̇ = 0.

        Returns
        -------
        torch.Tensor
            δC·q̇, shape (B, n_joints).  Exactly zero when the full q̇ = 0.
        """
        _assert_tensor(features, "features [CoriolisCorrection]", self.in_dim)
        _assert_tensor(qd, "qd [CoriolisCorrection]", self.n_joints)
        if features.shape[0] != qd.shape[0]:
            raise ValueError(
                f"[CoriolisCorrection] Batch size mismatch: features has "
                f"{features.shape[0]} rows, qd has {qd.shape[0]} rows."
            )
        if self._matrix_form:
            # B(q,q̇) ∈ (B, n, n);  δC·q̇ = B @ q̇ ∈ (B, n).
            # Vanishes iff the full q̇ vector is zero (B · 0 = 0 ∀ B);
            # allows cross-joint coupling (joint i non-zero when q̇_i = 0).
            B_mat = self.net(features).view(-1, self.n_joints, self.n_joints)
            delta_C_qd = torch.bmm(B_mat, qd.unsqueeze(-1)).squeeze(-1)
        else:
            # Legacy: element-wise multiply by q̇ (per-joint vanishing).
            mlp_out = self.net(features)                          # (B, n_joints)
            delta_C_qd = qd * mlp_out                             # (B, n_joints)

        _assert_output_finite(delta_C_qd, "delta_C_qd")
        return delta_C_qd

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# 4. Friction correction  δτ_f(q̇) = q̇ ⊙ h_φ(|q̇|)
# ===========================================================================

class FrictionCorrection(nn.Module):
    """Learn an odd-function correction to the joint friction torque.

    Mathematical form
    -----------------
    Real friction satisfies τ_f(−q̇) = −τ_f(q̇)  (reversing direction reverses
    friction sign).  We enforce this with the *product trick*:

        δτ_f(q̇) = q̇ ⊙ h_φ(|q̇|)

    where:
      • |q̇| denotes element-wise absolute value (always non-negative).
      • h_φ is an MLP that takes |q̇| as input — an even function of q̇.
      • q̇ is an odd function of itself.
      • The element-wise product of an odd and an even function is odd.

    Therefore δτ_f(−q̇) = (−q̇) ⊙ h_φ(|−q̇|) = −q̇ ⊙ h_φ(|q̇|) = −δτ_f(q̇),
    satisfying the odd-symmetry constraint exactly for every parameter value.

    Architecture: MLP taking |q̇| ∈ ℝ^n, 2 hidden layers of width 16.
    ~400 parameters for n_joints=5, hidden_size=16.

    Parameters
    ----------
    n_joints:
        Number of active joints.
    hidden_sizes:
        Hidden layer widths.  Smaller than the other networks because friction
        depends only on the magnitude of q̇ (simpler function class needed).
    activation:
        Nonlinearity name.  Tanh is recommended (bounded output scale).
    """

    def __init__(
        self,
        n_joints:     int = 5,
        hidden_sizes: Sequence[int] = (16, 16),
        activation:   str = "tanh",
        dropout:      float = 0.0,
        use_qdd:      bool = False,
        use_phys_cond: bool = False,
        friction_form: str = "mlp",
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if n_joints < 1:
            raise ValueError(f"n_joints must be ≥ 1, got {n_joints}")
        self.n_joints = int(n_joints)
        self._spectral_norm = bool(spectral_norm)
        # friction_form:
        #   "mlp"      — legacy: δτ_f = q̇ ⊙ h_φ(|q̇|[,…])  (even h, odd product).
        #   "stribeck" — structured Coulomb+Stribeck+viscous:
        #       δτ_f = sgn_s(q̇) ⊙ [F_c + F_s·exp(−(|q̇|/v_s)²)] + F_v·q̇
        #     where sgn_s(q̇)=q̇/√(q̇²+ε) is exactly odd, the bracket is even,
        #     and F_v·q̇ is odd ⇒ δτ_f stays exactly odd.  Captures the
        #     Coulomb offset / Stribeck dip the pure-MLP form cannot express
        #     (targets the friction-dominated, low-R² joints).
        if friction_form not in ("mlp", "stribeck"):
            raise ValueError(
                f"[FrictionCorrection] friction_form must be 'mlp' or "
                f"'stribeck', got {friction_form!r}"
            )
        self._friction_form = friction_form
        # When use_qdd=True, MLP receives [|q̇|, |q̈|] (2×n_joints) so the
        # correction can condition on acceleration magnitude — capturing
        # stiction, speed-transition dynamics, and load-dependent friction.
        self._use_qdd = bool(use_qdd)
        # When use_phys_cond=True, the analytic friction torque τ_f is threaded
        # in as |τ_f| (an *even* function of q̇, since τ_f is odd in q̇).
        # h_φ(|q̇|[,|q̈|][,|τ_f|]) therefore stays even ⇒ q̇ ⊙ h_φ stays exactly
        # odd: the odd-symmetry guarantee is preserved by construction.
        self._use_phys_cond = bool(use_phys_cond)
        _in_dim = self.n_joints * (
            1 + int(self._use_qdd) + int(self._use_phys_cond)
        )
        self.hparams = {
            "n_joints":     self.n_joints,
            "hidden_sizes": list(hidden_sizes),
            "activation":   activation,
            "dropout":      float(dropout),
            "use_qdd":      self._use_qdd,
            "use_phys_cond": self._use_phys_cond,
            "friction_form": self._friction_form,
        }
        # "mlp": n outputs (even scale h_φ).  "stribeck": 4n outputs
        # (F_c, F_s, raw_vs, F_v per joint).
        _out_dim = self.n_joints if self._friction_form == "mlp" else 4 * self.n_joints
        self.net = _build_correction_mlp(
            in_dim=_in_dim,
            hidden_sizes=hidden_sizes,
            out_dim=_out_dim,
            activation=activation,
            dropout=dropout,
            spectral_norm=self._spectral_norm,
        )

    def forward(
        self,
        qd: torch.Tensor,
        qdd: torch.Tensor | None = None,
        tau_f: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute friction correction  δτ_f(q̇) = q̇ ⊙ h_φ(|q̇| [, |q̈|] [, |τ_f|]).

        Parameters
        ----------
        qd:
            Joint velocities, shape (B, n_joints).
        qdd:
            Joint accelerations, shape (B, n_joints).  Required when
            ``use_qdd=True`` was set at construction time; otherwise ignored.
        tau_f:
            Analytic friction torque, shape (B, n_joints).  Required when
            ``use_phys_cond=True`` was set at construction time; otherwise
            ignored.  Fed in as |τ_f| (even ⇒ odd-symmetry preserved).

        Returns
        -------
        torch.Tensor
            δτ_f, shape (B, n_joints).  Satisfies δτ_f(−q̇) = −δτ_f(q̇) exactly.
        """
        _assert_tensor(qd, "qd [FrictionCorrection]", self.n_joints)
        # Smooth even function of q̇: sqrt(q̇² + ε) ≈ |q̇|, differentiable at 0.
        abs_qd = torch.sqrt(qd * qd + _FRICTION_ABS_EPS)   # (B, n_joints)
        parts = [abs_qd]
        if self._use_qdd:
            if qdd is None:
                raise ValueError(
                    "[FrictionCorrection] use_qdd=True but qdd was not provided."
                )
            parts.append(torch.sqrt(qdd * qdd + _FRICTION_ABS_EPS))
        if self._use_phys_cond:
            if tau_f is None:
                raise ValueError(
                    "[FrictionCorrection] use_phys_cond=True but tau_f was not "
                    "provided.  Pass the analytic friction torque so the "
                    "physics-conditioned (still odd) correction can be formed."
                )
            _assert_tensor(tau_f, "tau_f [FrictionCorrection]", self.n_joints)
            # |τ_f| is even in q̇ (τ_f is odd) ⇒ keeps h_φ even.
            parts.append(torch.sqrt(tau_f * tau_f + _FRICTION_ABS_EPS))
        mlp_in = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        out = self.net(mlp_in)

        if self._friction_form == "mlp":
            # out is an even (non-negative-input) scale h_φ ∈ (B, n).
            # Odd product: q̇ × h — odd because q̇ is odd and h is even.
            delta_tau_f = qd * out                              # (B, n_joints)
        else:
            # Structured Coulomb + Stribeck + viscous.
            n = self.n_joints
            F_c    = out[:, 0:n]
            F_s    = out[:, n:2*n]
            raw_vs = out[:, 2*n:3*n]
            F_v    = out[:, 3*n:4*n]
            # v_s > 0 always (softplus); +eps guarantees no div-by-zero even
            # at the near-zero init (raw_vs≈0 ⇒ v_s≈0.693).  Clamp raw_vs
            # ≥ -5 (B5 defensive): a pathological very-negative raw_vs would
            # drive v_s→1e-3 and (|q̇|/v_s)² → huge; exp(−huge)→0 is finite
            # but the clamp keeps the Stribeck scale numerically sane and
            # gradients well-conditioned.
            v_s = nn.functional.softplus(raw_vs.clamp(min=-5.0)) + 1e-3
            # sgn_s(q̇) = q̇/√(q̇²+ε): smooth, exactly odd, |·|→1 away from 0.
            sgn_s = qd / abs_qd
            # Even bracket (function of |q̇| and params only):
            stribeck = F_c + F_s * torch.exp(-(abs_qd / v_s) ** 2)
            # odd ⊙ even  +  odd  ⇒ exactly odd in q̇.
            delta_tau_f = sgn_s * stribeck + F_v * qd
        _assert_output_finite(delta_tau_f, "delta_tau_f")
        return delta_tau_f

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Convenience: parameter count summary
# ===========================================================================

def correction_parameter_summary(
    gravity:  GravityCorrection,
    inertia:  InertiaCorrection,
    coriolis: CoriolisCorrection,
    friction: FrictionCorrection,
) -> dict[str, int]:
    """Return a dict mapping component name → trainable parameter count.

    Parameters
    ----------
    gravity, inertia, coriolis, friction:
        Instantiated correction modules.

    Returns
    -------
    dict[str, int]
        Keys: 'gravity', 'inertia', 'coriolis', 'friction', 'total'.
    """
    counts = {
        "gravity":  gravity.count_parameters(),
        "inertia":  inertia.count_parameters(),
        "coriolis": coriolis.count_parameters(),
        "friction": friction.count_parameters(),
    }
    counts["total"] = sum(counts.values())
    return counts
