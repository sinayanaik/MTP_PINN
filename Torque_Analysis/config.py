"""
All constants, file paths, and tuneable parameters in one place.
config.py

Structure
---------
  HARDWARE CONSTANTS  — servo specs, conversion factors (never change)
  PATHS               — file locations relative to project root
  MASS CALIBRATION    — global density scale + per-joint servo mass corrections
  SIGNAL PROCESSING   — differentiation method and window sizes
  FRICTION MODEL      — Coulomb + viscous parameters per joint
  PLOT                — figure size / DPI

Tuning guide
------------
To improve model accuracy:

1. Mass calibration (MASS_SCALE / EXTRA_MASSES)
   Run calibrate_mass.py on slow quasi-static trajectories where inertial
   torques are negligible (τ ≈ g(q)).  Adjust α until RNEA/Load ≈ 1.0 for
   J2/J3 (gravity-dominated joints).

2. Stall torque (STALL_TORQUE_PER_JOINT)
   If joints J4/J5 use a different servo model (e.g., STS3032 → 14.8 kgf·cm),
   update indices 3 and 4.  An incorrect stall torque scales τ_load by a
   constant factor, which shows up as RNEA/Load ≠ 1.0 even for J2/J3.
   Diagnostic: if RNEA/Load for J4 is ~0.30 and you suspect STS3032,
   the true ratio would be 0.30 × (30/14.8) = 0.61 — still off, but much
   closer, suggesting combined stall-torque + friction mismatch.

3. Friction model (COULOMB_NM, VISCOUS_NM, FRICTION_EPS)
   Run calibrate_friction.py --bulk.  For joints where the load is
   friction-dominated (J1: yaw, J4/J5: distal wrist) and NRMSE > 60 %,
   check for:
     • Asymmetric friction (c⁺ ≠ c⁻): the gearbox may resist more in one
       direction.  Add per-direction Coulomb terms.
     • Larger FRICTION_EPS: a wider tanh transition models stiction better.

4. Smooth window (SMOOTH_WINDOW)
   Larger window → more noise rejection in qdd → better RNEA for fast motion.
   Smaller window → better temporal resolution for slow trajectories.
   Rule of thumb: window / fb_hz ≈ 50–100 ms is a good starting point.
"""

import os
import logging
import numpy as np

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"
LOG_DATE_FMT = "%H:%M:%S"

def setup_logging(level=None):
    """Call once at script entry point."""
    logging.basicConfig(
        level=level or LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FMT,
    )

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_JSON = os.path.join(
    PROJECT_ROOT,
    "raw_samples",
    "circle_r65mm_xz_cx66cyn265cz275_quintic_poly_ctrlmax_fbmax_001.json",
)

XACRO_PATH = os.path.join(
    PROJECT_ROOT,
    "robot_description", "urdf", "kikobot.xacro",
)

# ============================================================
# ROBOT HARDWARE CONSTANTS  (never change unless hardware changes)
# ============================================================
DOF           = 6
ACTIVE_JOINTS = 5           # joints 0-4 are actuated; joint 5 (tool) is passive

# Torque constant: τ (N·m) = KT × I (A)
# KT = 11.0 is a rough estimate; calibrated value from calibrate_mass.py is
# absorbed into MASS_SCALE instead (KT effectively cancels in load-register path).
KT            = 11.0        # N·m / A  (used only by torque_from_current — legacy)

# Scalar fallback (legacy) — superseded by STALL_TORQUE_PER_JOINT below.
STALL_TORQUE  = 30.0        # kgf·cm

NOM_VOLTAGE   = 12.0        # V — rated supply voltage (τ_stall is quoted at this V)
KGCM_TO_NM   = 0.09807      # kgf·cm → N·m  (= 9.807 / 1000 * 10 = 0.09807)

# Per-joint stall torques (kgf·cm).
#
# These are READ by torque_from_load_raw() to convert the dimensionless load
# register into N·m.  Getting these wrong introduces a proportional scaling
# error in τ_load — the "ground truth" — which then shows up as a biased
# RNEA/Load ratio across all trajectories.
#
# Feetech servo stall torques (at 12 V nominal):
#   STS3215 →  30.0 kgf·cm  (shoulder/elbow, large actuators)
#   STS3032 →  14.8 kgf·cm  (wrist/distal joints, smaller actuators)
#
# How to detect a wrong stall torque
# ------------------------------------
# Run `python3 calibrate_mass.py --bulk` and read the STALL TORQUE DIAGNOSTIC.
# For each joint the diagnostic computes:
#
#     τ_stall_inferred = τ_stall_assumed × (α_global / α_j)
#
# If τ_stall_inferred ≈ 14.8 kgf·cm while the assumed value is 30.0, the
# joint almost certainly uses an STS3032 servo.
#
# Bulk calibration result (2026-03-28, 124 files, 470 K samples):
#   J4 (q-index 3): α_j = 0.211, α_global = 0.093
#                   → τ_stall_inferred = 30.0 × (0.093/0.211) = 13.2 kgf·cm ≈ 14.8
#                   → Strong evidence J4 uses STS3032
#   J5 (q-index 4): gravity signal too weak for reliable inference;
#                   physically likely STS3032 given joint size/load requirements
#
# Effect of correcting J4 stall torque: RNEA/Load_J4 goes from 0.25 → ~0.51
# (τ_load halved; RNEA unchanged). After re-running bulk calibration and
# friction, further improvement to ~0.80+ is expected.
#
# Joint 6 (index 5) is passive — its stall torque is unused in practice.
# J4 (q-index 3) confirmed STS3032 by stall torque diagnostic (bulk, 2026-03-28):
#   α_j4 = 0.211, α_global = 0.093 → τ_stall_inferred = 30.0 × (0.093/0.211) = 13.2 ≈ 14.8
# J5 (q-index 4): physically likely STS3032 but gravity signal too weak to confirm.
#   Left at 30.0 until confirmed by direct measurement or improved calibration.
STALL_TORQUE_PER_JOINT = np.array([30.0, 30.0, 30.0, 14.8, 14.8, 30.0])

# Clip bounds for numerical differentiation output.
# These are generous safety margins — they should NEVER be hit for normal motion.
# If they are hit, it indicates timestamp repair failure or encoder wrap-around.
MAX_JOINT_VEL_RAD  = 10.0    # rad/s — rated no-load speed of the servo
VEL_CLIP           = 100.0   # rad/s — 10× rated speed (catches differentiation spikes)
ACC_CLIP           = 1000.0  # rad/s² — physically implausible above this value

# ============================================================
# CALIBRATION PARAMETER LOADING
#
# Parameters are loaded from calibration_params.json (project root).
# That file is written by calibrate_mass.py and calibrate_friction.py.
# If the file doesn't exist yet, the hardcoded defaults below are used.
#
# This means you never need to hand-edit config.py after running a
# calibration script — just run the script and the JSON updates.
# ============================================================

def _load_calib_json() -> dict:
    """
    Silently load calibration_params.json.
    Returns {"mass": dict|None, "friction": dict|None}.
    """
    try:
        from Torque_Analysis.calibration_io import load_calibration
        return load_calibration()
    except Exception:
        return {"mass": None, "friction": None}

_CALIB = _load_calib_json()


# ============================================================
# MASS CALIBRATION
#
# The URDF was built with a nominal (solid) density.  The actual robot is
# 3D-printed in PLA at ~70% infill, so effective density ≈ 875 kg/m³ vs
# URDF nominal 7800 kg/m³ → α = 875/7800 ≈ 0.112.
#
# Calibration procedure (calibrate_mass.py, WLS closed form):
#   Minimise Σ ||τ_load − α·τ_rnea_unit||² over quasi-static samples
#   Solution: α* = (aᵀb) / (aᵀa)  where a = τ_rnea_unit, b = τ_load
#   Result: α = 0.111893  (≈ 0.112)
#
# Limitation of global α
# ----------------------
# The servo motors (metal, ~60 g each) contribute FIXED mass that does NOT
# scale with the printed structure's density.  For distal joints (J3–J5)
# the servo mass dominates the link mass, so:
#   • α underestimates effective inertia for distal links
#   • RNEA/Load < 1 is expected for J3–J5 even after optimal α
#
# Fix: add servo masses as EXTRA_MASSES after applying α.
# Example: if servos are not modelled in the URDF at all, add 0.060 kg at
# each joint where a servo is physically attached.
# Currently None because a partial correction is already absorbed in α.
# ============================================================
# ── Hardcoded defaults (used if calibration_params.json is absent) ──
_DEFAULT_MASS_SCALE   = 0.111893
_DEFAULT_EXTRA_MASSES = None

# ── Load from JSON if available ──────────────────────────────────────
_mp = _CALIB.get("mass") or {}
MASS_SCALE   = float(_mp.get("mass_scale",   _DEFAULT_MASS_SCALE))
EXTRA_MASSES = _mp.get("extra_masses",        _DEFAULT_EXTRA_MASSES)

# ============================================================
# SIGNAL PROCESSING
# ============================================================

# Window length for Savitzky-Golay differentiation (must be odd, > SAVGOL_POLYORDER).
#
# Larger window → more noise rejection in q̈ → better RNEA for high-noise data.
# Smaller window → better temporal resolution for fast, highly dynamic motion.
#
# At 240 Hz feedback:  SMOOTH_WINDOW=25 → ~104 ms window  (recommended)
# At 480 Hz feedback:  SMOOTH_WINDOW=25 → ~52 ms window
#
# Window=11 (old default) gave NRMSE ~43-47% on J2/J3 because the second
# derivative estimate from a narrow window was too noisy to reconstruct
# the inertial term M(q)·q̈ accurately.
# Window=25 reduces noise by √(25/11) ≈ 1.5× in the acceleration estimate.
SMOOTH_WINDOW    = 25       # SG window for qd, qdd (was 11)

TORQUE_SMOOTH    = 21       # window for current-based torque smoothing (legacy)

# Differentiation method: "savgol" (recommended) or "gradient".
# "gradient" uses np.gradient (central differences), which is sensitive to
# any non-uniform spacing and produces ±inf spikes at zero-gap timestamps.
# "savgol" uses the MEAN timestep as the uniform spacing, making it robust
# to the occasional 1 ns nudge from fix_timestamps.
DIFF_METHOD      = "savgol"

# Polynomial order for the Savitzky-Golay filter.
# Must satisfy: SMOOTH_WINDOW > SAVGOL_POLYORDER.
# Order 3 (cubic) is a good default: fits velocity profiles well and avoids
# overfitting encoder quantisation noise.
# Order 5 can help for highly dynamic trajectories but requires window ≥ 7.
SAVGOL_POLYORDER = 3

# ============================================================
# FRICTION MODEL — calibrated via calibrate_friction.py --bulk
#
# Model:  τ_f(q̇) = c · tanh(q̇ / ε)  +  v · q̇
#
# Source: bulk sweep over 124 trajectories, 470 K samples
# Method: ε-sweep + constrained least-squares (physical bounds c ≥ 0, v ≥ 0)
#
# Per-joint observations (from diagnostic run 28-03-2026):
#   J1 (yaw):       friction_rms ≈ 0.191 N·m vs load_rms ≈ 0.210 N·m → 91% amplitude
#                   NRMSE = 63 % (amplitude close, but phase mismatch remains)
#   J2 (shoulder):  friction_rms ≈ 0.234 N·m (42 % of RNEA magnitude)
#   J3 (elbow):     friction_rms ≈ 0.172 N·m (47 % of RNEA magnitude)
#   J4 (wrist):     friction_rms ≈ 0.123 N·m vs load_rms ≈ 0.213 N·m → 58%
#   J5 (wrist roll):friction_rms ≈ 0.148 N·m vs load_rms ≈ 0.207 N·m → 71%
#                   NRMSE = 72 % → Coulomb may need to be ~0.27 N·m for better fit
#   J6 (tool):      passive — friction unused
#
# Improvement notes:
#   • J5 Coulomb could be increased (0.2151 → ~0.27) based on load_rms − rnea_rms
#   • Asymmetric Coulomb (c⁺ ≠ c⁻) could help if gearbox has backlash asymmetry
#   • Re-run calibrate_friction.py after fixing the single-pass derivative
#     (the old calibration was done with the double-pass derivative, so
#     qd estimates were phase-shifted — a new calibration may yield better params)
# ============================================================
# ── Hardcoded defaults (used if calibration_params.json is absent) ──
_DEFAULT_COULOMB_NM   = np.array([0.2472, 0.2873, 0.2403, 0.1806, 0.2151, 0.0])
_DEFAULT_VISCOUS_NM   = np.array([0.0,    0.3,    0.0,    0.051,  0.0042, 0.0])
_DEFAULT_FRICTION_EPS = 0.0628

# ── Load from JSON if available ──────────────────────────────────────
_fp = _CALIB.get("friction") or {}
if _fp.get("coulomb_nm"):
    COULOMB_NM = np.array(_fp["coulomb_nm"])
else:
    COULOMB_NM = _DEFAULT_COULOMB_NM.copy()

if _fp.get("viscous_nm"):
    VISCOUS_NM = np.array(_fp["viscous_nm"])
else:
    VISCOUS_NM = _DEFAULT_VISCOUS_NM.copy()

FRICTION_EPS = float(_fp.get("friction_eps", _DEFAULT_FRICTION_EPS))

# ============================================================
# PLOT
# ============================================================
FIG_WIDTH  = 15
FIG_HEIGHT = 5
DPI        = 120
