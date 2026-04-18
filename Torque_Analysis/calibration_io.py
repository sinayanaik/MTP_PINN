"""
Calibration parameter persistence.
calibration_io.py

All calibration scripts write their results here instead of requiring the user
to manually edit config.py.  config.py reads from this file at import time.

File location
-------------
Torque_Analysis/calibration_params.json   (same directory as this module)

JSON structure
--------------
{
  "schema_version": "1.1",
  "mass": {
    "current": {
      "mass_scale":         float,        -- global density scale α
      "extra_masses":       null | dict,  -- {joint_idx: kg} lumped additions
      "total_mass_kg":      float,        -- α × unscaled URDF total mass
      "source_file":        str,          -- which log was used for calibration
      "n_samples":          int,
      "convention":         str,          -- sign convention that gave best fit
      "per_joint_scales":   list[float|null], -- per-joint optimal α (diagnostic)
      "calibrated_at":      ISO 8601 str
    },
    "history": [ ...older "current" dicts... ]
  },
  "friction": {
    "current": {
      "coulomb_nm":   list[float],   -- Coulomb friction per joint [N·m]
      "viscous_nm":   list[float],   -- viscous drag per joint [N·m·s/rad]
      "friction_eps": float,         -- tanh transition width [rad/s]
      "bulk":         bool,          -- True if calibrated on all files
      "n_samples":    int,
      "source_info":  str,           -- human-readable description
      "rms_old":      list[float],   -- per-joint residual RMS before update
      "rms_new":      list[float],   -- per-joint residual RMS after update
      "calibrated_at": ISO 8601 str
    },
    "history": [ ...older "current" dicts... ]
  }
}

Usage
-----
# In calibration scripts:
from Torque_Analysis.calibration_io import save_mass_params, save_friction_params

# In config.py (auto-called at import):
from Torque_Analysis.calibration_io import load_calibration
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Location: always next to Torque_Analysis/ inside the project root
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
CALIB_FILE  = os.path.join(_THIS_DIR, "calibration_params.json")

_SCHEMA_VERSION = "1.1"

# Maximum number of historical entries to retain per category.
# Older entries beyond this limit are dropped (FIFO).
MAX_HISTORY = 20


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_raw() -> dict:
    """Load the raw JSON dict from disk, or return an empty skeleton."""
    if not os.path.exists(CALIB_FILE):
        return {
            "schema_version": _SCHEMA_VERSION,
            "mass":     {"current": None, "history": []},
            "friction": {"current": None, "history": []},
        }
    try:
        with open(CALIB_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Could not read %s: %s — using empty skeleton", CALIB_FILE, e)
        return {
            "schema_version": _SCHEMA_VERSION,
            "mass":     {"current": None, "history": []},
            "friction": {"current": None, "history": []},
        }


def _save_raw(data: dict) -> None:
    """Write the JSON dict back to disk atomically."""
    data["schema_version"] = _SCHEMA_VERSION
    tmp = CALIB_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, CALIB_FILE)   # atomic on POSIX
        logger.info("Saved calibration params → %s", CALIB_FILE)
    except OSError as e:
        logger.error("Failed to save calibration params: %s", e)
        raise


def _push_history(section: dict) -> None:
    """
    Move section["current"] into section["history"] (front of list).
    Trims history to MAX_HISTORY entries.
    """
    if section.get("current") is not None:
        section["history"].insert(0, section["current"])
        section["history"] = section["history"][:MAX_HISTORY]


# ──────────────────────────────────────────────────────────────
# Public load API  (called by config.py at import time)
# ──────────────────────────────────────────────────────────────

def load_calibration() -> dict:
    """
    Load current calibration parameters from disk.

    Returns a dict with keys 'mass' and 'friction', each containing the
    "current" sub-dict (or None if never calibrated).

    This function is safe to call even if the file doesn't exist yet —
    it returns None for uncalibrated sections, allowing config.py to
    fall back to its built-in defaults.
    """
    raw  = _load_raw()
    mass = raw.get("mass", {}).get("current")

    # JSON serialises dict keys as strings; convert back to int so that
    # build_pinocchio_model can index model.inertias[joint_idx] correctly.
    if mass and isinstance(mass.get("extra_masses"), dict):
        mass["extra_masses"] = {
            int(k): v for k, v in mass["extra_masses"].items()
        }

    return {
        "mass":     mass,
        "friction": raw.get("friction", {}).get("current"),
    }


# ──────────────────────────────────────────────────────────────
# Save: mass calibration
# ──────────────────────────────────────────────────────────────

def save_mass_params(
        mass_scale: float,
        total_mass_kg: float,
        source_file: str,
        n_samples: int,
        convention: str,
        per_joint_scales: list,
        extra_masses: dict | None = None,
) -> None:
    """
    Persist mass calibration results.

    Called at the end of calibrate_mass.py after the best-fit α is found.
    Moves the previous "current" entry to history before writing the new one.

    Parameters
    ----------
    mass_scale       : α — the global density scale factor
    total_mass_kg    : α × unscaled URDF total mass
    source_file      : basename of the JSON log used for calibration
    n_samples        : number of samples used
    convention       : e.g. "-direction × raw"
    per_joint_scales : per-joint optimal α list (None for joints with
                       insufficient gravity signal)
    extra_masses     : dict {joint_idx: extra_kg} or None
    """
    data = _load_raw()
    _push_history(data["mass"])

    data["mass"]["current"] = {
        "mass_scale":       round(float(mass_scale), 8),
        "extra_masses":     extra_masses,
        "total_mass_kg":    round(float(total_mass_kg), 6),
        "source_file":      os.path.basename(source_file),
        "n_samples":        int(n_samples),
        "convention":       convention,
        "per_joint_scales": [
            round(float(s), 6) if s is not None and not _is_nan(s) else None
            for s in per_joint_scales
        ],
        "calibrated_at":    _now_iso(),
    }

    _save_raw(data)
    print(f"\n  ✓ Mass calibration saved → {CALIB_FILE}")
    print(f"    mass_scale = {mass_scale:.6f}  "
          f"({len(data['mass']['history'])} previous entries in history)")


# ──────────────────────────────────────────────────────────────
# Save: friction calibration
# ──────────────────────────────────────────────────────────────

def save_friction_params(
        coulomb_nm: list,
        viscous_nm: list,
        friction_eps: float,
        bulk: bool,
        n_samples: int,
        source_info: str,
        rms_old: list | None = None,
        rms_new: list | None = None,
) -> None:
    """
    Persist friction calibration results.

    Called at the end of calibrate_friction.py after synthesis.
    Moves the previous "current" entry to history before writing the new one.

    Parameters
    ----------
    coulomb_nm   : per-joint Coulomb friction [N·m]
    viscous_nm   : per-joint viscous drag [N·m·s/rad]
    friction_eps : tanh transition width [rad/s]
    bulk         : True if all files were used (--bulk flag)
    n_samples    : total number of samples processed
    source_info  : human-readable description of the data source
    rms_old      : per-joint residual RMS before update
    rms_new      : per-joint residual RMS after update
    """
    data = _load_raw()
    _push_history(data["friction"])

    data["friction"]["current"] = {
        "coulomb_nm":   [round(float(c), 6) for c in coulomb_nm],
        "viscous_nm":   [round(float(v), 6) for v in viscous_nm],
        "friction_eps": round(float(friction_eps), 6),
        "bulk":         bool(bulk),
        "n_samples":    int(n_samples),
        "source_info":  source_info,
        "rms_old":      [round(float(r), 6) for r in rms_old] if rms_old else None,
        "rms_new":      [round(float(r), 6) for r in rms_new] if rms_new else None,
        "calibrated_at": _now_iso(),
    }

    _save_raw(data)
    print(f"\n  ✓ Friction calibration saved → {CALIB_FILE}")
    print(f"    COULOMB_NM   = {[round(float(c), 4) for c in coulomb_nm]}")
    print(f"    VISCOUS_NM   = {[round(float(v), 4) for v in viscous_nm]}")
    print(f"    FRICTION_EPS = {friction_eps:.4f}")
    print(f"    ({len(data['friction']['history'])} previous entries in history)")


# ──────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────

def _is_nan(x) -> bool:
    try:
        import math
        return math.isnan(x)
    except (TypeError, ValueError):
        return False


def print_calibration_status() -> None:
    """Print a brief summary of the current calibration state."""
    params = load_calibration()

    print("\nCalibration status")
    print("─" * 50)

    mp = params["mass"]
    if mp:
        print(f"  Mass:     α={mp['mass_scale']:.6f}  "
              f"mass={mp['total_mass_kg']:.4f} kg  "
              f"[{mp['calibrated_at']}]")
    else:
        print("  Mass:     ← using config.py defaults (never calibrated)")

    fp = params["friction"]
    if fp:
        c = [f"{v:.4f}" for v in fp["coulomb_nm"]]
        v = [f"{v:.4f}" for v in fp["viscous_nm"]]
        print(f"  Friction: ε={fp['friction_eps']:.4f}  "
              f"bulk={fp['bulk']}  "
              f"[{fp['calibrated_at']}]")
        print(f"    COULOMB = {c}")
        print(f"    VISCOUS = {v}")
    else:
        print("  Friction: ← using config.py defaults (never calibrated)")
