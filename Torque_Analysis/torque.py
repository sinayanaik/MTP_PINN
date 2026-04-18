"""
Three independent torque estimates + friction model.
torque.py

Torque sources
--------------
(a) torque_from_load  — sensor measurement from the servo load register.
    This is the "ground truth" we try to predict.  Formula:
        τ_load = (load_register × 0.001) × τ_stall × (V / V_nom)
    where load_register is in 0.1 % units (1000 = 100 % stall torque).

(b) torque_from_urdf  — analytical inverse dynamics via RNEA.
    Recursive Newton-Euler Algorithm gives the torque needed to produce
    the observed motion (q, qd, qdd) against gravity:
        τ = M(q)·q̈  +  C(q,q̇)·q̇  +  g(q)
    where:
        M(q)         — joint-space mass matrix  (captures inertia)
        C(q,q̇)·q̇   — Coriolis + centripetal forces
        g(q)         — gravity compensation torques

(c) torque_from_load − torque_from_urdf ≡ residual
    Contains friction, backlash, unmodelled compliance, and sensor noise.

(d) torque_friction   — explicit Coulomb + viscous model added on top of RNEA:
        τ_f = c · tanh(q̇ / ε)  +  v · q̇
    tanh replaces the discontinuous sign() so the model is differentiable
    (needed for gradient-based calibration).

Full model prediction:
        τ_model = τ_RNEA + τ_friction

Diagnostic notes (from bulk analysis of 124 trajectories, 470 K samples)
Calibration: α=0.093 (mass), ε=0.041, bulk friction via calibrate_friction.py
------------------------------------------------------------------------
J1 (yaw):  RNEA/Load ≈ 0.055, NRMSE ≈ 51 %.
    Yaw rotation about the vertical axis has near-zero gravity torque.
    The load is almost entirely dry friction (stiction + viscous).
    The tanh model is an approximation; stiction onset (zero-crossing
    behaviour) cannot be captured without a Stribeck or LuGre model.
    Viscous hits the calibration cap (0.30 N·m·s/rad) for this joint.

J2 (shoulder): RNEA/Load ≈ 0.787, NRMSE ≈ 35 %.
    Per-joint α (0.096) ≈ global α (0.093), so the gravity term is
    correctly calibrated.  The RNEA/Load < 1 on mixed trajectories comes
    from the INERTIAL term M(q)·q̈ being underestimated: servo motor
    masses (~65 g each at J3–J5) are not in the URDF and do not scale
    with the PLA density correction α.  Adding them via EXTRA_MASSES
    would improve dynamic (fast-trajectory) accuracy.

J3 (elbow): RNEA/Load ≈ 0.844, NRMSE ≈ 34 %.  Same story as J2.

J4 (wrist): RNEA/Load ≈ 0.254, NRMSE ≈ 58 %.
    Per-joint α from bulk calibration is 0.211 vs global 0.093 (ratio
    2.27).  The ratio matches 30.0/14.8 = 2.03 (STS3215 vs STS3032 stall
    torque), indicating J4 uses an STS3032 servo whose stall torque
    (14.8 kgf·cm) is lower than the assumed STS3215 (30 kgf·cm).
    Fixing C.STALL_TORQUE_PER_JOINT[3] = 14.8 should bring RNEA/Load
    to ~0.50, and after re-calibrating α and friction, to ~0.80+.
    Run `python3 calibrate_mass.py --bulk` to see the stall torque
    diagnostic output.

J5 (wrist roll): RNEA/Load ≈ 0.035, NRMSE ≈ 64 %.
    Almost no link mass beyond J5; load is dominated by friction.
    Per-joint α is unreliable (gravity signal too weak).  If J5 also uses
    STS3032 (likely, same joint size as J4), updating
    STALL_TORQUE_PER_JOINT[4] = 14.8 would halve τ_load and let the
    friction calibration find a cleaner match.

J6 (tool, passive): not driven — ignored in RNEA.
"""

from __future__ import annotations

import logging
import numpy as np

from . import config as C
from .utils import (numerical_velocity, numerical_acceleration, smooth,
                    velocity_and_acceleration_from_pos)

logger = logging.getLogger(__name__)


# ==================================================================
# (a)  Current-based torque  (legacy — load-register is preferred)
# ==================================================================
def torque_from_current(current: np.ndarray, load: np.ndarray,
                        signed: bool = False) -> np.ndarray:
    """
    Estimate torque from motor phase current (mA) → N·m.

    τ = KT × I   where KT is the motor torque constant (N·m/A).

    The sign is ambiguous from current alone (current is always positive
    in many servo protocols), so we use np.sign(load) from the load
    register to recover direction when signed=False.

    This is LESS accurate than torque_from_load because:
    • The torque constant KT is temperature-dependent.
    • Current-based sensing integrates winding resistance losses.
    Use torque_from_load (load register) as the primary sensor reading.
    """
    if signed:
        tau = (current / 1000.0) * C.KT * C.KGCM_TO_NM
    else:
        tau = np.sign(load) * np.abs(current / 1000.0) * C.KT * C.KGCM_TO_NM
    return tau


def torque_from_current_smoothed(current: np.ndarray, load: np.ndarray,
                                 signed: bool = False,
                                 window: int = None) -> np.ndarray:
    """Current-based torque with moving-average smoothing to reduce PWM ripple."""
    if window is None:
        window = C.TORQUE_SMOOTH
    tau = torque_from_current(current, load, signed)
    return smooth(tau, window)


# ==================================================================
# (b)  Load-register torque  ← PRIMARY sensor for ground-truth τ
# ==================================================================
def torque_from_load_raw(load: np.ndarray,
                         voltage: np.ndarray,
                         stall_torque_per_joint: np.ndarray = None) -> np.ndarray:
    """
    Convert the servo load register to torque in the SERVO frame (N·m).

    Physics
    -------
    The Feetech servo reports output shaft torque as a percentage of its
    rated stall torque, in 0.1 % units (so a value of 500 = 50.0 %).
    The stall torque itself is proportional to supply voltage (permanent
    magnet motor, torque ∝ I ∝ V/R at stall), so:

        τ_servo = (load × 0.001) × τ_stall(kgf·cm) × (V / V_nom) × KGCM_TO_NM

    The voltage correction is per-sample because the supply sags under load.

    CRITICAL: τ_stall must match the actual servo model per joint.
        Feetech STS3215 → τ_stall = 30.0 kgf·cm  (shoulder/elbow joints)
        Feetech STS3032 → τ_stall = 14.8 kgf·cm  (wrist joints if smaller)
    Using the wrong τ_stall causes a proportional error in τ_load.
    See C.STALL_TORQUE_PER_JOINT in config.py.

    Parameters
    ----------
    load                   : (N, njoints)  raw load register values (0.1% units)
    voltage                : (N, njoints)  per-joint supply voltage (V)
    stall_torque_per_joint : (njoints,)    stall torques in kgf·cm
    """
    if stall_torque_per_joint is None:
        nj = load.shape[1] if load.ndim == 2 else 1
        stall_torque_per_joint = C.STALL_TORQUE_PER_JOINT[:nj]

    # load * 0.001: convert 0.1% units → fraction of stall torque
    load_frac = load * 0.001
    # Voltage correction: stall torque scales linearly with supply voltage
    v_scale = voltage / C.NOM_VOLTAGE
    # Broadcast stall_torque_per_joint (njoints,) over the time axis (N,)
    tau = load_frac * stall_torque_per_joint * v_scale * C.KGCM_TO_NM
    return tau


def torque_from_load(load: np.ndarray,
                     voltage: np.ndarray,
                     joint_map: list = None,
                     stall_torque_per_joint: np.ndarray = None) -> np.ndarray:
    """
    Load-register torque converted to URDF frame (N·m).

    The servo frame and the URDF frame may differ in sign for each joint
    depending on how the servo is physically mounted (gearbox direction).
    The direction sign and zero-position offset are stored in joint_map,
    calibrated by calibrate_mass.py.

    Frame convention:
        τ_urdf = −direction × τ_servo

    The negation is because positive URDF torque is defined as the torque
    that RESISTS joint motion (reaction torque), whereas the servo reports
    the torque it APPLIES.

    Parameters
    ----------
    stall_torque_per_joint : optional per-joint override.  If None, uses
        C.STALL_TORQUE_PER_JOINT.  Pass a custom array to test different
        servo model assumptions without editing config.py.
    """
    tau_servo = torque_from_load_raw(load, voltage, stall_torque_per_joint)

    if joint_map is not None:
        direction = np.array([jm["direction"] for jm in joint_map])
        tau_urdf = -direction * tau_servo
        return tau_urdf
    else:
        return tau_servo


# ==================================================================
# (c)  URDF analytical (Pinocchio RNEA)
# ==================================================================
def build_pinocchio_model(xacro_path: str = None,
                          mass_scale: float = None,
                          extra_masses: dict = None):
    """
    Load xacro → URDF XML → Pinocchio model + data object.

    Mass calibration
    ----------------
    The URDF is generated from a CAD model that was likely created at a
    nominal density (e.g., solid PLA or steel).  The actual robot is
    3D-printed at reduced infill, so all link masses must be scaled down.

    mass_scale (α):
        Calibrated by matching RNEA gravity torques to measured load torques
        during quasi-static poses (slow motion, negligible inertial terms).
        α ≈ 0.112 corresponds to PLA at ~70% infill (density 875 vs 7800 kg/m³).

        For a uniform density change, inertia tensors scale proportionally
        with mass (I = α·I_nominal), so BOTH mass AND inertia are scaled.

    extra_masses (per-joint lumped additions):
        The servo motors themselves are metal and unaffected by the PLA
        density factor.  Use this dict to add servo masses at the joints
        where they are physically located.
        Example: {1: 0.060, 2: 0.060}  → 60 g servo at joints 1 and 2.

    NOTE: The global α is optimal for the structural (PLA) links.  Distal
    joints (J3–J5) are dominated by servo mass (metal, not PLA), so the
    effective density is HIGHER than α predicts.  This is a known source
    of model error and can be corrected by per-joint extra_masses.

    Parameters
    ----------
    xacro_path   : path to the .xacro robot description file
    mass_scale   : α — multiplicative scale for all link masses and inertias
    extra_masses : {joint_index: extra_mass_kg} — lumped masses to add
                   after scaling (e.g., servo motors not in the CAD model)

    Returns
    -------
    model : pin.Model  — Pinocchio kinematic/dynamic model
    data  : pin.Data   — pre-allocated data object for RNEA calls
    nq    : int        — number of actuated degrees of freedom
    """
    import xacro
    import pinocchio as pin

    if xacro_path is None:
        xacro_path = C.XACRO_PATH
    if mass_scale is None:
        mass_scale = C.MASS_SCALE

    try:
        urdf_xml = xacro.process_file(xacro_path).toxml()
    except Exception as e:
        raise RuntimeError(
            f"Failed to process XACRO file '{xacro_path}': {e}"
        ) from e

    try:
        model = pin.buildModelFromXML(urdf_xml)
    except Exception as e:
        raise RuntimeError(
            f"Failed to build Pinocchio model from URDF: {e}"
        ) from e

    # Apply global density scale.
    # For a body with uniform density ρ:   mass ∝ ρ·V,   I ∝ ρ·V·r²
    # → both scale linearly with α = ρ_actual / ρ_nominal.
    if mass_scale != 1.0:
        for i in range(model.njoints):
            model.inertias[i].mass    *= mass_scale
            model.inertias[i].inertia *= mass_scale

    # Add lumped extra masses for components not in the CAD model.
    # Only mass is updated (inertia change is negligible for small point masses
    # when the link's existing inertia dominates).
    if extra_masses is not None:
        for joint_idx, extra_kg in extra_masses.items():
            if 0 <= joint_idx < model.njoints:
                model.inertias[joint_idx].mass += extra_kg

    data = model.createData()

    total_mass = sum(model.inertias[i].mass for i in range(model.njoints))
    logger.info("Pinocchio model: nq=%d, total_mass=%.4f kg, mass_scale=%.6f",
                model.nq, total_mass, mass_scale)

    return model, data, model.nq


def torque_from_urdf(model, data, q: np.ndarray,
                     t: np.ndarray,
                     smooth_window: int = None) -> tuple:
    """
    Full inverse dynamics via the Recursive Newton-Euler Algorithm (RNEA).

    RNEA computes the joint torques required to produce the observed motion:

        τ = M(q)·q̈  +  C(q,q̇)·q̇  +  g(q)

    where:
        M(q)       — (nq × nq) joint-space mass/inertia matrix
        C(q,q̇)·q̇ — Coriolis and centripetal terms
        g(q)       — gravity compensation torques

    Pinocchio implements RNEA as a two-pass O(n) algorithm:
        Forward pass:  propagate kinematics (positions, velocities,
                        accelerations) from root to leaves.
        Backward pass: accumulate forces and compute joint torques
                        from leaves back to root.

    Derivative strategy
    -------------------
    Both qd and qdd are computed from the SAME Savitzky-Golay polynomial
    fit of q (single-pass), avoiding the double-filtering phase error that
    occurs when differentiating qd a second time.  See utils.py for details.

    Parameters
    ----------
    model         : Pinocchio model (from build_pinocchio_model)
    data          : Pinocchio data  (model.createData())
    q             : (N, nq)  measured joint positions in radians
    t             : (N,)     timestamps in seconds
    smooth_window : SG window length (default C.SMOOTH_WINDOW)

    Returns
    -------
    tau : (N, nq)  RNEA torques in N·m
    qd  : (N, nq)  joint velocities in rad/s
    qdd : (N, nq)  joint accelerations in rad/s²
    """
    import pinocchio as pin

    if smooth_window is None:
        smooth_window = C.SMOOTH_WINDOW

    # Single-pass: both derivatives from the same polynomial fit of q.
    # This is the key fix for the 43-47 % NRMSE on J2/J3.
    qd, qdd = velocity_and_acceleration_from_pos(q, t, smooth_window)

    N = q.shape[0]
    tau = np.zeros((N, model.nq))
    for i in range(N):
        # pin.rnea is O(n) in the number of joints.
        # It uses model.gravity (set to [0, 0, -9.81] by Pinocchio default,
        # matching the robot's base frame where Z is up).
        tau[i] = pin.rnea(model, data, q[i], qd[i], qdd[i])

    return tau, qd, qdd


def torque_gravity_only(model, data, q: np.ndarray) -> np.ndarray:
    """
    Compute pure gravity compensation torques: τ_g = g(q).

    Equivalent to RNEA with q̇ = 0 and q̈ = 0, which zeros out both
    the inertial term M(q)·q̈ and the Coriolis term C(q,q̇)·q̇.
    The result is the torque each joint motor must supply to hold the
    arm stationary against gravity.

    Used for:
    • Diagnosing mass calibration (τ_gravity should match τ_load at slow speeds)
    • Splitting RNEA into gravity vs inertial contributions
    """
    import pinocchio as pin

    N = q.shape[0]
    zero = np.zeros(model.nv)    # pre-allocate once, reused every iteration
    tau_g = np.zeros((N, model.nq))
    for i in range(N):
        tau_g[i] = pin.rnea(model, data, q[i], zero, zero)
    return tau_g


# ==================================================================
# (d)  Friction model — smooth Coulomb + viscous
# ==================================================================
def _smooth_sign(x: np.ndarray, eps: float = None) -> np.ndarray:
    """
    Differentiable approximation of sign(x) via hyperbolic tangent.

        smooth_sign(x) = tanh(x / ε)

    Why tanh instead of sign()
    --------------------------
    The classical dry-friction model uses sign(q̇), which is discontinuous
    at q̇ = 0.  This creates two problems:
        1. Numerical chatter in simulation (rapid sign flips).
        2. Non-differentiability, which breaks gradient-based calibration
           (L-BFGS-B, Adam, etc.) because ∂τ/∂c is undefined at q̇ = 0.

    tanh(x/ε) is smooth everywhere and converges to sign(x) as ε → 0:
        • At |q̇| >> ε:  tanh ≈ ±1  → pure Coulomb behaviour
        • At |q̇| ≈ ε:  linear ramp → models the stiction-to-sliding
                         transition (Stribeck-like, though simplified)
        • At |q̇| = 0:  tanh = 0    → zero friction at rest (no stiction)

    ε (FRICTION_EPS in config.py):
        Transition half-width in rad/s.  ε = 0.0628 rad/s corresponds to
        ~0.6 rpm.  Increase ε for a smoother/wider transition; decrease it
        to recover sharper Coulomb behaviour.

    Parameters
    ----------
    x   : array of velocities in rad/s
    eps : transition width ε (default C.FRICTION_EPS)
    """
    if eps is None:
        eps = C.FRICTION_EPS
    return np.tanh(x / eps)


def torque_friction(qd: np.ndarray,
                    coulomb: np.ndarray = None,
                    viscous: np.ndarray = None,
                    eps: float = None) -> np.ndarray:
    """
    Coulomb + viscous friction model:

        τ_f(q̇) = c · tanh(q̇ / ε)  +  v · q̇

    where:
        c  — Coulomb (dry) friction amplitude  [N·m]   (C.COULOMB_NM)
        v  — viscous drag coefficient           [N·m·s/rad] (C.VISCOUS_NM)
        ε  — tanh transition width              [rad/s] (C.FRICTION_EPS)

    Physical interpretation
    -----------------------
    • The Coulomb term c·tanh(q̇/ε) models gear-tooth and seal friction
      that is approximately constant in magnitude regardless of speed.
      It opposes the direction of motion (sign of q̇).
    • The viscous term v·q̇ models oil/grease drag that grows linearly
      with speed.  For the gearbox joints (J1, J2) this can be significant.
    • Together they represent the LuGre model's steady-state approximation.

    Calibration hint
    ----------------
    Run calibrate_friction.py --bulk for per-joint identification.
    For joints where load is dominated by friction (J1, J4, J5) and NRMSE
    is still high, consider:
        • Asymmetric Coulomb: c⁺ ≠ c⁻  (gearbox asymmetry)
        • Higher ε for softer stiction onset

    Parameters
    ----------
    qd      : (N, nq)  joint velocities in rad/s
    coulomb : (nq,)    Coulomb friction per joint in N·m
    viscous : (nq,)    viscous friction per joint in N·m·s/rad
    eps     : tanh transition width in rad/s
    """
    nq = qd.shape[1]
    if coulomb is None:
        coulomb = C.COULOMB_NM[:nq]
    if viscous is None:
        viscous = C.VISCOUS_NM[:nq]
    if eps is None:
        eps = C.FRICTION_EPS

    # Broadcast coulomb/viscous (nq,) across the time axis (N,)
    return coulomb * _smooth_sign(qd, eps) + viscous * qd
