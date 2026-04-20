"""
Standalone physics module for Kikobot 6-DOF robot arm.

Provides:
  - Robot hardware constants and calibrated parameters
  - Timestamp repair (monotonicity + outlier-dt + optional uniform resampling)
  - Savitzky-Golay smoothing and differentiation helpers
  - Torque conversion (load register -> Nm), friction model
  - RNEA inverse dynamics via Pinocchio (lazy import)

Dependencies: numpy, scipy.  Pinocchio and xacro are lazy-imported only when
RNEA functions are called.
"""

from __future__ import annotations

import logging
import os
import json
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# =============================================================================
# ROBOT HARDWARE CONSTANTS
# =============================================================================

DOF           = 6
ACTIVE_JOINTS = 5

NOM_VOLTAGE   = 12.0
KGCM_TO_NM    = 0.09807

# Per-joint stall torques (kgf-cm)
# J0-J2, J5: STS3215 -> 30.0;  J3 (wrist): STS3032 -> 14.8;  J4: 30.0
STALL_TORQUE_PER_JOINT = np.array([30.0, 30.0, 30.0, 14.8, 30.0, 30.0])

VEL_CLIP  = 100.0   # rad/s  -- clip for numerical outliers
ACC_CLIP  = 1000.0   # rad/s^2

SMOOTH_WINDOW    = 25
SAVGOL_POLYORDER = 3

# =============================================================================
# CALIBRATION
# =============================================================================

_DEFAULT_MASS_SCALE   = 0.09310315   # PLA ~70% infill vs URDF nominal density
_DEFAULT_EXTRA_MASSES = None
_DEFAULT_COULOMB_NM   = np.array([0.134975, 0.278199, 0.201313, 0.088112, 0.203864, 0.0])
_DEFAULT_VISCOUS_NM   = np.array([0.3,      0.3,      0.245417, 0.040191, 0.046918, 0.0])
_DEFAULT_FRICTION_EPS = 0.040469     # rad/s -- tanh transition width


def load_calibration_params(calib_json_path: str | None = None) -> dict:
    """Load calibrated physics parameters from JSON, falling back to defaults."""
    if calib_json_path is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(
            this_dir, "..", "Torque_Analysis", "calibration_params.json"
        )
        calib_json_path = os.path.normpath(candidate)

    params = {
        "mass_scale":   _DEFAULT_MASS_SCALE,
        "extra_masses": _DEFAULT_EXTRA_MASSES,
        "coulomb_nm":   _DEFAULT_COULOMB_NM.copy(),
        "viscous_nm":   _DEFAULT_VISCOUS_NM.copy(),
        "friction_eps": _DEFAULT_FRICTION_EPS,
    }

    if not os.path.exists(calib_json_path):
        logger.info("calibration_params.json not found -- using hardcoded defaults")
        return params

    try:
        with open(calib_json_path, "r") as f:
            raw = json.load(f)
        mp = (raw.get("mass", {}) or {}).get("current") or {}
        fp = (raw.get("friction", {}) or {}).get("current") or {}
        if mp.get("mass_scale") is not None:
            params["mass_scale"] = float(mp["mass_scale"])
        if mp.get("extra_masses") is not None:
            params["extra_masses"] = {int(k): v for k, v in mp["extra_masses"].items()}
        if fp.get("coulomb_nm"):
            params["coulomb_nm"] = np.array(fp["coulomb_nm"])
        if fp.get("viscous_nm"):
            params["viscous_nm"] = np.array(fp["viscous_nm"])
        if fp.get("friction_eps") is not None:
            params["friction_eps"] = float(fp["friction_eps"])
        logger.info("Loaded calibration from %s", calib_json_path)
    except Exception as e:
        logger.warning("Could not load calibration JSON (%s) -- using defaults", e)

    return params


_CALIB = load_calibration_params()
MASS_SCALE   = _CALIB["mass_scale"]
EXTRA_MASSES = _CALIB["extra_masses"]
COULOMB_NM   = _CALIB["coulomb_nm"]
VISCOUS_NM   = _CALIB["viscous_nm"]
FRICTION_EPS = _CALIB["friction_eps"]

_PHYSICS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(_PHYSICS_DIR))
XACRO_PATH = os.path.join(_PROJ_ROOT, "robot_description", "urdf", "kikobot.xacro")


# =============================================================================
# TIMESTAMP REPAIR
# =============================================================================

@dataclass
class TimestampReport:
    """Diagnostic report from fix_timestamps."""
    t_fixed:         np.ndarray = field(repr=False)
    n_nonmonotonic:  int   = 0
    n_outlier_dt:    int   = 0
    dt_cv_before:    float = 0.0
    dt_cv_after:     float = 0.0
    was_resampled:   bool  = False
    median_dt:       float = 0.0

    def to_dict(self) -> dict:
        """Serialisable summary (no arrays)."""
        return {
            "n_nonmonotonic": self.n_nonmonotonic,
            "n_outlier_dt":   self.n_outlier_dt,
            "dt_cv_before":   round(self.dt_cv_before, 6),
            "dt_cv_after":    round(self.dt_cv_after, 6),
            "was_resampled":  self.was_resampled,
            "median_dt_ms":   round(self.median_dt * 1000, 4),
        }


def fix_timestamps(
        t: np.ndarray,
        outlier_factor: float = 3.0,
        resample_cv_threshold: float = 0.05,
) -> TimestampReport:
    """
    Repair timestamps in three stages.

    1. Monotonicity: interpolate over dt <= 0 regions.
    2. Outlier dt: flag steps where dt deviates from median(dt) by more than
       ``outlier_factor`` and re-interpolate those regions.
    3. Uniform resampling: if CV(dt) still exceeds ``resample_cv_threshold``,
       resample to uniform spacing at median sample rate.

    Returns TimestampReport with repaired timestamps and diagnostics.
    """
    t_fixed = t.copy().astype(np.float64)
    n = len(t_fixed)

    if n < 3:
        med = float(np.diff(t_fixed).mean()) if n > 1 else 0.0
        return TimestampReport(t_fixed=t_fixed, median_dt=med)

    # --- Stage 1: monotonicity repair ---
    dt = np.diff(t_fixed)
    bad_mono = dt <= 0
    n_nonmonotonic = int(bad_mono.sum())

    if n_nonmonotonic > 0:
        good_mask = np.ones(n, dtype=bool)
        good_mask[np.where(bad_mono)[0] + 1] = False
        good_idx = np.where(good_mask)[0]
        if len(good_idx) < 2:
            t_fixed = np.linspace(float(t[0]), float(t[-1]), n)
        else:
            t_fixed = np.interp(np.arange(n), good_idx, t_fixed[good_idx])
        for k in range(1, n):
            if t_fixed[k] <= t_fixed[k - 1]:
                t_fixed[k] = t_fixed[k - 1] + 1e-9

    # --- Stage 2: outlier dt repair ---
    dt = np.diff(t_fixed)
    median_dt = float(np.median(dt))
    dt_mean = float(np.mean(dt))
    dt_cv_before = float(np.std(dt) / dt_mean) if dt_mean > 0 else 0.0
    n_outlier_dt = 0

    if median_dt > 0:
        lo = median_dt / outlier_factor
        hi = median_dt * outlier_factor
        outlier_mask = (dt < lo) | (dt > hi)
        n_outlier_dt = int(outlier_mask.sum())

        if n_outlier_dt > 0:
            good_mask = np.ones(n, dtype=bool)
            bad_indices = np.where(outlier_mask)[0] + 1
            bad_indices = bad_indices[bad_indices < n]
            good_mask[bad_indices] = False
            good_idx = np.where(good_mask)[0]
            if len(good_idx) >= 2:
                t_fixed = np.interp(np.arange(n), good_idx, t_fixed[good_idx])
            for k in range(1, n):
                if t_fixed[k] <= t_fixed[k - 1]:
                    t_fixed[k] = t_fixed[k - 1] + 1e-9

    # --- Stage 3: optional uniform resampling ---
    dt = np.diff(t_fixed)
    dt_mean = float(np.mean(dt))
    dt_cv_after = float(np.std(dt) / dt_mean) if dt_mean > 0 else 0.0
    was_resampled = False

    if dt_cv_after > resample_cv_threshold and n > 2:
        t_fixed = np.linspace(t_fixed[0], t_fixed[-1], n)
        was_resampled = True
        dt = np.diff(t_fixed)
        dt_mean = float(np.mean(dt))
        dt_cv_after = float(np.std(dt) / dt_mean) if dt_mean > 0 else 0.0
        logger.info("Timestamps resampled to uniform spacing (CV was %.4f)", dt_cv_after)

    median_dt = float(np.median(np.diff(t_fixed)))

    if n_nonmonotonic > 0 or n_outlier_dt > 0:
        logger.warning(
            "Timestamp repair: %d non-monotonic, %d outlier dt, CV %.4f -> %.4f%s",
            n_nonmonotonic, n_outlier_dt, dt_cv_before, dt_cv_after,
            ", resampled" if was_resampled else "",
        )

    return TimestampReport(
        t_fixed=t_fixed,
        n_nonmonotonic=n_nonmonotonic,
        n_outlier_dt=n_outlier_dt,
        dt_cv_before=dt_cv_before,
        dt_cv_after=dt_cv_after,
        was_resampled=was_resampled,
        median_dt=median_dt,
    )


# =============================================================================
# SAVITZKY-GOLAY HELPERS
# =============================================================================

def validated_sg_window(signal_len: int, requested_win: int, polyorder: int) -> int:
    """Return a valid (odd, > polyorder, <= signal_len) SG window."""
    win = requested_win if requested_win % 2 == 1 else requested_win + 1
    min_win = polyorder + 1
    if min_win % 2 == 0:
        min_win += 1
    win = max(win, min_win)
    max_win = signal_len if signal_len % 2 == 1 else signal_len - 1
    win = min(win, max_win)
    return win


def savgol_smooth(signal: np.ndarray, window_length: int, polyorder: int,
                  axis: int = 0) -> np.ndarray:
    """
    Savitzky-Golay smoothing (deriv=0).

    Validates window size automatically.  Returns a copy of the original
    signal if it is too short for the requested polynomial order.
    """
    n = signal.shape[axis]
    if n < polyorder + 2:
        return signal.copy()
    win = validated_sg_window(n, window_length, polyorder)
    return savgol_filter(signal, window_length=win, polyorder=polyorder,
                         deriv=0, axis=axis)


def sg_differentiate(
        q: np.ndarray,
        dt: float,
        window_length: int = SMOOTH_WINDOW,
        polyorder: int = SAVGOL_POLYORDER,
        mode: str = "interp",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity and acceleration from position via SG differentiation.

    Both qd (deriv=1) and qdd (deriv=2) come from the SAME polynomial fit,
    guaranteeing qdd = d/dt(qd) within the polynomial approximation.

    Parameters
    ----------
    q              : (N, nq) joint positions [rad]
    dt             : uniform timestep [s] (use median_dt from TimestampReport)
    window_length  : SG window
    polyorder      : SG polynomial order (forced >= 2 for non-trivial qdd)
    mode           : SG boundary mode

    Returns
    -------
    qd  : (N, nq) [rad/s]
    qdd : (N, nq) [rad/s^2]
    """
    N = q.shape[0]
    polyorder = max(polyorder, 2)
    win = validated_sg_window(N, window_length, polyorder)

    qd = savgol_filter(q, window_length=win, polyorder=polyorder,
                       deriv=1, delta=dt, axis=0, mode=mode)
    qdd = savgol_filter(q, window_length=win, polyorder=polyorder,
                        deriv=2, delta=dt, axis=0, mode=mode)

    qd  = np.clip(qd,  -VEL_CLIP,  VEL_CLIP)
    qdd = np.clip(qdd, -ACC_CLIP,  ACC_CLIP)
    qd  = np.nan_to_num(qd,  nan=0.0, posinf=0.0, neginf=0.0)
    qdd = np.nan_to_num(qdd, nan=0.0, posinf=0.0, neginf=0.0)

    return qd, qdd


def raw_derivatives(q: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute raw (unfiltered) derivatives via np.gradient.

    Used only for the ``raw_qd`` and ``raw_qdd`` CSV columns -- noisy but
    unprocessed for comparison.
    """
    qd  = np.gradient(q, t, axis=0)
    qdd = np.gradient(qd, t, axis=0)

    qd  = np.clip(qd,  -VEL_CLIP,  VEL_CLIP)
    qdd = np.clip(qdd, -ACC_CLIP,  ACC_CLIP)
    qd  = np.nan_to_num(qd,  nan=0.0, posinf=0.0, neginf=0.0)
    qdd = np.nan_to_num(qdd, nan=0.0, posinf=0.0, neginf=0.0)

    return qd, qdd


# =============================================================================
# COORDINATE CONVERSION
# =============================================================================

def ticks_to_radians(act_pos: np.ndarray, joint_map: list,
                     ticks_to_rad: float, dof: int = 6) -> np.ndarray:
    """
    Encoder ticks -> joint angles in radians (URDF frame).

        q_i = direction_i * (ticks_i - ticks_center_i) * ticks_to_rad
    """
    N = act_pos.shape[0]
    q = np.zeros((N, dof))
    for i, jm in enumerate(joint_map):
        q[:, i] = jm["direction"] * (act_pos[:, i] - jm["ticks_center"]) * ticks_to_rad
    return q


# =============================================================================
# TORQUE FUNCTIONS
# =============================================================================

def torque_from_load(load: np.ndarray, voltage: np.ndarray,
                     joint_map: list | None = None,
                     stall_torque_per_joint: np.ndarray | None = None) -> np.ndarray:
    """
    Servo load register -> torque in URDF frame [Nm].

        tau_servo = (load * 0.001) * stall(kgf-cm) * (V / V_nom) * KGCM_TO_NM
        tau_urdf  = -direction * tau_servo
    """
    nj = load.shape[1] if load.ndim == 2 else 1
    if stall_torque_per_joint is None:
        stall_torque_per_joint = STALL_TORQUE_PER_JOINT[:nj]

    tau_servo = load * 0.001 * stall_torque_per_joint * (voltage / NOM_VOLTAGE) * KGCM_TO_NM

    if joint_map is not None:
        direction = np.array([jm["direction"] for jm in joint_map])
        return -direction * tau_servo
    return tau_servo


def torque_friction(qd: np.ndarray,
                    coulomb: np.ndarray | None = None,
                    viscous: np.ndarray | None = None,
                    eps: float | None = None) -> np.ndarray:
    """
    Smooth Coulomb + viscous friction:  tau_f = c * tanh(qd/eps) + v * qd
    """
    nq = qd.shape[1] if qd.ndim == 2 else len(qd)
    if coulomb is None:
        coulomb = COULOMB_NM[:nq]
    if viscous is None:
        viscous = VISCOUS_NM[:nq]
    if eps is None:
        eps = FRICTION_EPS
    return coulomb * np.tanh(qd / eps) + viscous * qd


# =============================================================================
# PINOCCHIO RNEA
# =============================================================================

def build_pinocchio_model(xacro_path: str | None = None,
                          mass_scale: float | None = None,
                          extra_masses: dict | None = None):
    """
    Load xacro -> URDF -> Pinocchio model + data.

    Applies mass density scale alpha and optional extra lumped masses.
    Returns (model, data, nq).
    """
    import xacro
    import pinocchio as pin

    if xacro_path is None:
        xacro_path = XACRO_PATH
    if mass_scale is None:
        mass_scale = MASS_SCALE

    urdf_xml = xacro.process_file(xacro_path).toxml()
    model    = pin.buildModelFromXML(urdf_xml)

    if mass_scale != 1.0:
        for i in range(model.njoints):
            model.inertias[i].mass    *= mass_scale
            model.inertias[i].inertia *= mass_scale

    if extra_masses is not None:
        for joint_idx, extra_kg in extra_masses.items():
            if 0 <= joint_idx < model.njoints:
                model.inertias[joint_idx].mass += extra_kg

    data       = model.createData()
    total_mass = sum(model.inertias[i].mass for i in range(model.njoints))
    logger.info("Pinocchio model: nq=%d, total_mass=%.4f kg, alpha=%.6f",
                model.nq, total_mass, mass_scale)
    return model, data, model.nq


def compute_rnea_decomposition(
        model, data,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        n_active: int = ACTIVE_JOINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full rigid-body inverse-dynamics decomposition via RNEA.

        tau_g    = RNEA(q, 0, 0)           -- gravity
        tau_M    = RNEA(q, 0, qdd) - tau_g -- inertia M(q)qdd
        tau_C    = RNEA(q, qd, 0)  - tau_g -- Coriolis C(q,qd)qd
        tau_rnea = tau_g + tau_M + tau_C   -- full rigid-body

    Returns (tau_rnea, tau_gravity, tau_M, tau_C), each (N, n_active).
    """
    import pinocchio as pin

    N   = q.shape[0]
    mnq = model.nq
    nv  = model.nv

    def _pad(arr: np.ndarray) -> np.ndarray:
        cols = arr.shape[1]
        if cols >= mnq:
            return arr[:, :mnq]
        return np.hstack([arr, np.zeros((N, mnq - cols), dtype=arr.dtype)])

    q_p   = _pad(q)
    qd_p  = _pad(qd)
    qdd_p = _pad(qdd)
    zeros = np.zeros(nv, dtype=np.float64)

    tau_rnea    = np.zeros((N, mnq))
    tau_gravity = np.zeros((N, mnq))
    tau_M       = np.zeros((N, mnq))
    tau_C       = np.zeros((N, mnq))

    for i in range(N):
        qi   = q_p[i].astype(np.float64)
        qdi  = qd_p[i].astype(np.float64)
        qddi = qdd_p[i].astype(np.float64)

        tau_gravity[i] = pin.rnea(model, data, qi, zeros, zeros)
        tau_rnea[i]    = pin.rnea(model, data, qi, qdi,   qddi)
        tau_M[i]       = pin.rnea(model, data, qi, zeros, qddi) - tau_gravity[i]
        tau_C[i]       = pin.rnea(model, data, qi, qdi,   zeros) - tau_gravity[i]

    na = min(n_active, mnq)
    return (
        tau_rnea[:, :na].astype(np.float32),
        tau_gravity[:, :na].astype(np.float32),
        tau_M[:, :na].astype(np.float32),
        tau_C[:, :na].astype(np.float32),
    )
