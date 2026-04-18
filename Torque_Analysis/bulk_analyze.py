#!/usr/bin/env python3
"""
Bulk torque analysis pipeline.

Output layout
─────────────
infer_torque/
└── DDMMYYYY_eps{eps}_c{cmin}-{cmax}_v{vmin}-{vmax}/   ← one folder per run
    ├── trajectory_plots/       ← one PNG per trajectory (Load/RNEA/RNEA+Fric)
    │   ├── {run_id}.png
    │   └── ...
    ├── global_plots/           ← cross-run summary plots
    │   ├── rnea_ratio_violin.png
    │   ├── nrmse_violin.png
    │   ├── residual_rms_boxplot.png
    │   ├── accuracy_by_shape.png
    │   ├── accuracy_by_traj_type.png
    │   ├── load_vs_rnea_scatter.png
    │   ├── accuracy_vs_radius.png
    │   ├── error_hist_global.png
    │   └── error_hist_by_shape_J*.png
    ├── torque_data.npz         ← all torque arrays concatenated (~470 K samples)
    ├── global_summary.json     ← rich metadata (shape-wise + joint aggregates)
    └── metadata.txt            ← human-readable run report

Usage
─────
    cd ~/Desktop/MTP_PINN/Torque_Analysis
    python3 bulk_analyze.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob
import os
import re
import json
import time
import logging
import traceback
import numpy as np

from Torque_Analysis import config as C
from Torque_Analysis.data_loader import load_log
from Torque_Analysis.utils import ticks_to_radians
from Torque_Analysis.torque import (
    torque_from_load,
    build_pinocchio_model,
    torque_from_urdf,
    torque_gravity_only,
    torque_friction,
)
from Torque_Analysis.plots import set_headless

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
RAW_DIR      = os.path.join(C.PROJECT_ROOT, "raw_samples")
BASE_OUT_DIR = os.path.join(C.PROJECT_ROOT, "infer_torque")
DOF          = C.DOF  # 6


def get_run_folder_name() -> str:
    """
    Descriptive run-folder name: DDMMYYYY_eps{eps}_c{cmin}-{cmax}_v{vmin}-{vmax}

    Example: 27032025_eps0.0628_c0.181-0.287_v0.000-0.300
    """
    date_str = time.strftime("%d%m%Y")
    eps      = C.FRICTION_EPS
    c_act    = C.COULOMB_NM[:5]   # active joints only
    v_act    = C.VISCOUS_NM[:5]
    return (
        f"{date_str}"
        f"_eps{eps:.4f}"
        f"_c{min(c_act):.3f}-{max(c_act):.3f}"
        f"_v{min(v_act):.3f}-{max(v_act):.3f}"
    )

# Fixed histogram bins shared across all runs and plots.
# 100 bins spanning ±2.5 N·m — covers the full residual range for this robot.
HIST_BINS       = np.linspace(-2.5, 2.5, 101)   # 101 edges → 100 bins
HIST_BIN_EDGES  = HIST_BINS.tolist()             # serialisable for JSON

KNOWN_SHAPES = [
    "regular_polygon", "sine_wave",  # multi-word first (longest match)
    "circle", "ellipse", "helix", "spiral", "triangle",
    "parabola", "lissajous", "rectangle", "square",
]
KNOWN_TRAJ_TYPES = ["quintic_poly", "cubic_poly", "ruckig", "trapezoidal", "linear"]


# ─────────────────────────────────────────────────────────────
# Filename metadata parser
# ─────────────────────────────────────────────────────────────

def parse_filename(fname: str) -> dict:
    """
    Extract structured metadata from a log filename.

    Pattern (approx):
      {shape}_{r_field}_{plane}_{center}_{traj_type}_{ctrl}_{fb}_{seq}.json

    Returns dict with keys:
      shape, radius_mm, plane, traj_type, fb_max, fb_hz, seq
    """
    stem = Path(fname).stem  # strip .json

    # --- shape (longest match first to handle multi-word names) ---
    shape = "unknown"
    for s in KNOWN_SHAPES:
        if stem.startswith(s + "_"):
            shape = s
            break

    # --- radius ---
    m = re.search(r"r(\d+)mm", stem)
    radius_mm = int(m.group(1)) if m else None

    # --- plane / orientation ---
    m = re.search(r"_(xz|yz|xy|az-?\d+el-?\d+)_", stem)
    plane = m.group(1) if m else "unknown"
    # Simplify tilted planes to "tilted"
    if plane.startswith("az"):
        plane = "tilted"

    # --- trajectory profile ---
    traj_type = "unknown"
    for t in KNOWN_TRAJ_TYPES:
        if t in stem:
            traj_type = t
            break

    # --- feedback rate ---
    m = re.search(r"fb(max|\d+hz)", stem, re.IGNORECASE)
    fb_str = m.group(1).lower() if m else "unknown"
    fb_max = (fb_str == "max")
    m2 = re.search(r"(\d+)hz", fb_str)
    fb_hz = int(m2.group(1)) if m2 else None

    # --- sequence number ---
    m = re.search(r"_(\d{3})$", stem)
    seq = int(m.group(1)) if m else None

    return {
        "shape":      shape,
        "radius_mm":  radius_mm,
        "plane":      plane,
        "traj_type":  traj_type,
        "fb_max":     fb_max,
        "fb_hz":      fb_hz,
        "seq":        seq,
    }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def safe_ratio(a, b):
    """Return a/b rounded, or None if b is tiny or result is extreme."""
    if b is None or abs(b) < 1e-4:
        return None
    r = a / b
    if abs(r) > 1000:
        return None
    return round(float(r), 4)


def safe_float(x):
    """Float with nan/inf guard."""
    if x is None:
        return None
    v = float(x)
    return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)


# ─────────────────────────────────────────────────────────────
# Per-run processing
# ─────────────────────────────────────────────────────────────

def process_one_file(json_path: str, model) -> tuple:
    """
    Run the full torque-estimation pipeline for one trajectory log file.

    Pipeline stages
    ---------------
    1. Load JSON → raw arrays (act_pos, load, voltage, timestamps)
    2. Encoder ticks → joint angles q in radians (URDF frame)
    3. SENSOR torque:  τ_load = load_register → N·m  (ground truth)
    4. RNEA torque:    τ_rnea = M(q)q̈ + C(q,q̇)q̇ + g(q)  via Pinocchio
       • q̇ and q̈ computed by single-pass SG differentiation of q
         (avoids the double-filtering phase error of the old two-step approach)
    5. GRAVITY torque: τ_grav = RNEA with q̇=0, q̈=0  (for diagnostics)
    6. FRICTION torque:τ_fric = c·tanh(q̇/ε) + v·q̇
    7. FULL MODEL:     τ_model = τ_rnea + τ_fric
    8. RESIDUAL:       τ_resid = τ_load − τ_model  (unmodelled dynamics)

    Returns
    -------
    summary : dict   — per-joint statistics (RMS, NRMSE, ratios, histograms)
    arrays  : dict   — raw (N, nq) torque arrays for plotting and npz saving
    """
    L, M, N = load_log(json_path)
    run_id = M["run_id"]
    nq     = model.nq

    # Stage 1 → 2: encoder ticks to joint angles in radians
    # direction sign and zero-point offset come from per-joint calibration
    act_rad = ticks_to_radians(
        L["act_pos"], M["joint_map"], M["ticks_to_rad"], C.DOF
    )
    q = act_rad[:, :nq]   # keep only actuated joints

    # Stage 3: sensor torque — load register → N·m (URDF frame)
    # This is the signal we are trying to predict with our model.
    pdata    = model.createData()
    tau_load = torque_from_load(L["load"], L["voltage"],
                                joint_map=M["joint_map"])

    # Stage 4: RNEA — single-pass SG differentiation inside torque_from_urdf
    # tau_urdf = M(q)q̈ + C(q,q̇)q̇ + g(q)   [N·m]
    tau_urdf, qd, qdd = torque_from_urdf(model, pdata, q, L["t"])

    # Stage 5: gravity-only (for diagnosing mass calibration)
    # tau_gravity = g(q) = RNEA with q̇=q̈=0
    tau_gravity = torque_gravity_only(model, pdata, q)

    # Stage 6: friction torque
    # tau_fric = c·tanh(q̇/ε) + v·q̇
    tau_fric = torque_friction(qd)

    # Stage 7: full model prediction
    # tau_model ≈ tau_load  (ideally)
    tau_model = tau_urdf + tau_fric

    # Stage 8: residual = what the model doesn't capture
    # Large residuals indicate: backlash, compliance, friction mismatch,
    # mass model error, or sensor noise.
    tau_residual = tau_load[:, :nq] - tau_model

    duration = float(L["t"][-1] - L["t"][0])

    # --- Per-joint statistics ---
    # NRMSE = residual_rms / load_rms  (lower = better model)
    #   < 0.20 → excellent  (model explains >80% of load variance in RMS sense)
    #   < 0.50 → acceptable
    #   > 0.50 → model is missing significant dynamics for this joint
    #
    # RNEA/Load ratio ≈ 1.0 → mass calibration is correct for this joint
    #   ratio << 1 → model under-predicts (mass too low, or stall torque too high)
    #   ratio >> 1 → model over-predicts (mass too high, or stall torque too low)
    joints = []
    for j in range(DOF):
        load_rms = safe_float(np.sqrt(np.mean(tau_load[:, j] ** 2)))
        js = {
            "joint":     j,
            "name":      f"revolute_{j+1}",
            "load_rms":  load_rms,
            "load_mean": safe_float(np.mean(tau_load[:, j])),
            "load_min":  safe_float(np.min(tau_load[:, j])),
            "load_max":  safe_float(np.max(tau_load[:, j])),
            "load_std":  safe_float(np.std(tau_load[:, j])),
        }

        if j < nq:
            rnea_rms   = float(np.sqrt(np.mean(tau_urdf[:, j] ** 2)))
            grav_rms   = float(np.sqrt(np.mean(tau_gravity[:, j] ** 2)))
            fric_rms   = float(np.sqrt(np.mean(tau_fric[:, j] ** 2)))
            model_rms  = float(np.sqrt(np.mean(tau_model[:, j] ** 2)))
            resid_rms  = float(np.sqrt(np.mean(tau_residual[:, j] ** 2)))
            resid_mean = float(np.mean(tau_residual[:, j]))

            # nrmse = residual_rms / load_rms
            # Interpretation: fraction of the load signal power NOT explained
            # by the model.  E.g. nrmse=0.43 means 43% unexplained residual.
            nrmse = safe_ratio(resid_rms, load_rms)

            js.update({
                "rnea_rms":       safe_float(rnea_rms),
                "gravity_rms":    safe_float(grav_rms),
                "friction_rms":   safe_float(fric_rms),
                "model_rms":      safe_float(model_rms),
                "residual_rms":   safe_float(resid_rms),
                "residual_mean":  safe_float(resid_mean),
                "residual_bias":  safe_float(abs(resid_mean) / max(load_rms, 1e-6)),
                "rnea_over_load": safe_ratio(rnea_rms,
                                             load_rms if load_rms is not None else 1e-9),
                "model_over_load": safe_ratio(model_rms,
                                              load_rms if load_rms is not None else 1e-9),
                "nrmse":          nrmse,   # residual RMS / load RMS (lower=better)
                "q_range_deg":    safe_float(np.degrees(q[:, j].max() - q[:, j].min())),
                "qd_max":         safe_float(np.abs(qd[:, j]).max()),
                "qd_mean":        safe_float(np.mean(np.abs(qd[:, j]))),
                "qdd_max":        safe_float(np.abs(qdd[:, j]).max()),
            })

        joints.append(js)

    # --- Per-joint error histograms (3 models, active joints only) ---
    # Stored as binned counts so they can be aggregated across runs without
    # keeping all individual samples in memory.
    #
    # Three error signals per joint j:
    #   err_rnea  = τ_load[:,j] − τ_urdf[:,j]   (RNEA-only residual)
    #   err_model = τ_load[:,j] − τ_model[:,j]  (RNEA+Friction residual)
    #   err_fric  = τ_load[:,j] − τ_fric[:,j]   (Friction-only residual)
    error_hists = {}
    for j in range(nq):
        err_rnea  = tau_load[:, j] - tau_urdf[:, j]
        err_model = tau_load[:, j] - tau_model[:, j]
        err_fric  = tau_load[:, j] - tau_fric[:, j]
        error_hists[j] = {
            "rnea":  np.histogram(err_rnea,  bins=HIST_BINS)[0].tolist(),
            "model": np.histogram(err_model, bins=HIST_BINS)[0].tolist(),
            "fric":  np.histogram(err_fric,  bins=HIST_BINS)[0].tolist(),
            "n":     int(len(err_rnea)),
        }

    # --- Run-level summary ---
    traj_meta = parse_filename(json_path)
    summary = {
        "run_id":      run_id,
        "label":       M["label"],
        "success":     M["success"],
        "N":           N,
        "duration_s":  round(duration, 3),
        "ctrl_hz":     M["ctrl_hz"],
        "fb_hz":       M["fb_hz"],
        "traj_meta":   traj_meta,
        "joints":      joints,
        "error_hists": error_hists,   # {joint_idx: {rnea, model, fric, n}}
    }

    # --- Raw arrays for plotting and bulk save ---
    arrays = {
        "run_id":      run_id,
        "t":           L["t"],
        "q":           q,
        "qd":          qd,
        "tau_load":    tau_load[:, :nq],
        "tau_rnea":    tau_urdf,
        "tau_fric":    tau_fric,
        "tau_model":   tau_model,
        "tau_residual": tau_residual,
    }

    return summary, arrays


# ─────────────────────────────────────────────────────────────
# Global summary JSON builder
# ─────────────────────────────────────────────────────────────

def _agg(values):
    """Aggregate a list of floats → {min, max, mean, median, std}."""
    if not values:
        return None
    a = np.array(values, dtype=float)
    return {
        "min":    round(float(np.min(a)), 4),
        "max":    round(float(np.max(a)), 4),
        "mean":   round(float(np.mean(a)), 4),
        "median": round(float(np.median(a)), 4),
        "std":    round(float(np.std(a)), 4),
        "n":      int(len(a)),
    }


def _collect_joint_values(summaries, joint_idx, key):
    return [s["joints"][joint_idx][key]
            for s in summaries
            if s["joints"][joint_idx].get(key) is not None]


def build_global_summary(summaries, errors, total_time, model):
    """Build the global summary JSON with rich metadata."""
    total_mass = sum(model.inertias[i].mass for i in range(model.njoints))

    gs = {
        "schema_version": "2.0",
        "config": {
            "mass_scale":              C.MASS_SCALE,
            "total_mass_kg":           round(float(total_mass), 4),
            "smooth_window":           C.SMOOTH_WINDOW,
            "diff_method":             C.DIFF_METHOD,
            "savgol_polyorder":        C.SAVGOL_POLYORDER,
            "friction_eps":            C.FRICTION_EPS,
            "coulomb_nm":              C.COULOMB_NM.tolist(),
            "viscous_nm":              C.VISCOUS_NM.tolist(),
            "stall_torque_per_joint":  C.STALL_TORQUE_PER_JOINT.tolist(),
            "nom_voltage":             C.NOM_VOLTAGE,
            "dof":                     C.DOF,
        },
        "processing": {
            "total_files":         len(summaries) + len(errors),
            "succeeded":           len(summaries),
            "failed":              len(errors),
            "total_time_s":        round(total_time, 1),
            "avg_time_per_file_s": round(total_time / max(len(summaries) + len(errors), 1), 2),
        },
    }

    if not summaries:
        gs["errors"] = errors
        return gs

    # ── dataset-level ────────────────────────────────────────
    all_N   = [s["N"] for s in summaries]
    all_dur = [s["duration_s"] for s in summaries]

    gs["dataset"] = {
        "total_runs":       len(summaries),
        "total_samples":    int(sum(all_N)),
        "total_duration_s": round(float(sum(all_dur)), 1),
        "samples_per_run":  _agg(all_N),
        "duration_per_run": _agg(all_dur),
    }

    # ── per-joint aggregate ───────────────────────────────────
    joint_agg = []
    for j in range(DOF):
        ratio_all  = _collect_joint_values(summaries, j, "rnea_over_load")
        mrat_all   = _collect_joint_values(summaries, j, "model_over_load")
        nrmse_all  = _collect_joint_values(summaries, j, "nrmse")
        load_all   = _collect_joint_values(summaries, j, "load_rms")
        resid_all  = _collect_joint_values(summaries, j, "residual_rms")

        n_good_rnea  = sum(1 for r in ratio_all  if 0.5 < r < 2.0)
        n_good_model = sum(1 for r in mrat_all   if 0.5 < r < 2.0)

        ja = {
            "joint":           j,
            "name":            f"revolute_{j+1}",
            "load_rms":        _agg(load_all),
            "residual_rms":    _agg(resid_all),
            "rnea_over_load":  {**(_agg(ratio_all) or {}),
                                "n_good_ratio": n_good_rnea,
                                "pct_good":     round(100 * n_good_rnea / max(len(ratio_all), 1), 1)},
            "model_over_load": {**(_agg(mrat_all) or {}),
                                "n_good_ratio": n_good_model,
                                "pct_good":     round(100 * n_good_model / max(len(mrat_all), 1), 1)},
            "nrmse":           _agg(nrmse_all),
        }
        joint_agg.append(ja)

    gs["joint_aggregate"] = joint_agg

    # ── by trajectory shape ───────────────────────────────────
    shapes = sorted({s["traj_meta"]["shape"] for s in summaries})
    shape_agg = {}
    for shape in shapes:
        subset = [s for s in summaries if s["traj_meta"]["shape"] == shape]
        shape_agg[shape] = {
            "n_runs": len(subset),
            "joints": {},
        }
        for j in range(C.DOF):
            ratio_all  = _collect_joint_values(subset, j, "rnea_over_load")
            mrat_all   = _collect_joint_values(subset, j, "model_over_load")
            nrmse_all  = _collect_joint_values(subset, j, "nrmse")
            shape_agg[shape]["joints"][f"j{j}"] = {
                "rnea_over_load_median": round(float(np.median(ratio_all)), 4) if ratio_all else None,
                "model_over_load_median": round(float(np.median(mrat_all)), 4) if mrat_all else None,
                "nrmse_median": round(float(np.median(nrmse_all)), 4) if nrmse_all else None,
            }

    gs["by_shape"] = shape_agg

    # ── by trajectory profile type ────────────────────────────
    traj_types = sorted({s["traj_meta"]["traj_type"] for s in summaries})
    traj_agg = {}
    for tt in traj_types:
        subset = [s for s in summaries if s["traj_meta"]["traj_type"] == tt]
        traj_agg[tt] = {
            "n_runs": len(subset),
            "joints": {},
        }
        for j in range(C.DOF):
            ratio_all  = _collect_joint_values(subset, j, "rnea_over_load")
            nrmse_all  = _collect_joint_values(subset, j, "nrmse")
            traj_agg[tt]["joints"][f"j{j}"] = {
                "rnea_over_load_median": round(float(np.median(ratio_all)), 4) if ratio_all else None,
                "nrmse_median": round(float(np.median(nrmse_all)), 4) if nrmse_all else None,
            }

    gs["by_traj_type"] = traj_agg

    # ── by radius ─────────────────────────────────────────────
    radii = sorted({s["traj_meta"]["radius_mm"]
                    for s in summaries
                    if s["traj_meta"]["radius_mm"] is not None})
    radius_agg = {}
    for r in radii:
        subset = [s for s in summaries if s["traj_meta"]["radius_mm"] == r]
        nrmse_per_joint = {
            f"j{j}": round(float(np.median(
                _collect_joint_values(subset, j, "nrmse") or [float("nan")]
            )), 4)
            for j in range(C.DOF)
        }
        radius_agg[str(r)] = {
            "n_runs": len(subset),
            "nrmse_median_per_joint": nrmse_per_joint,
        }

    gs["by_radius_mm"] = radius_agg

    # ── cumulative model quality ──────────────────────────────
    cum = {}
    for j in range(C.DOF):
        nrmse_all = _collect_joint_values(summaries, j, "nrmse")
        if nrmse_all:
            a = np.array(nrmse_all)
            cum[f"j{j}"] = {
                "pct_runs_nrmse_lt_50pct": round(100 * np.mean(a < 0.5), 1),
                "pct_runs_nrmse_lt_30pct": round(100 * np.mean(a < 0.3), 1),
                "pct_runs_nrmse_lt_20pct": round(100 * np.mean(a < 0.2), 1),
                "median_nrmse_pct":        round(float(np.median(a)) * 100, 1),
            }

    gs["model_quality"] = {
        "description": (
            "nrmse = residual_rms / load_rms (lower is better). "
            "nrmse < 0.5 means model explains >50% of load variance (in RMS sense)."
        ),
        "per_joint": cum,
    }

    # ── error histograms (global + by shape) ─────────────────
    # Accumulate bin counts across runs so the full error distribution
    # is available for plotting without re-loading individual files.
    n_bins = len(HIST_BINS) - 1
    models = ("rnea", "model", "fric")

    def _empty_hists(nq):
        return {j: {m: np.zeros(n_bins, dtype=np.int64) for m in models}
                for j in range(nq)}

    def _add_hists(acc, s):
        for j, jh in s.get("error_hists", {}).items():
            j = int(j)
            if j not in acc:
                acc[j] = {m: np.zeros(n_bins, dtype=np.int64) for m in models}
            for m in models:
                if m in jh:
                    acc[j][m] += np.array(jh[m], dtype=np.int64)

    # Global accumulation
    global_hists = {}
    for s in summaries:
        _add_hists(global_hists, s)

    # Per-shape accumulation
    shape_hists = {}
    for shape in shapes:
        shape_hists[shape] = {}
        for s in summaries:
            if s["traj_meta"]["shape"] == shape:
                _add_hists(shape_hists[shape], s)

    # Serialise (numpy int64 → plain list)
    def _ser(h):
        return {str(j): {m: h[j][m].tolist() for m in models} for j in h}

    gs["error_histograms"] = {
        "description": (
            "Pre-binned error counts for τ_load − τ_model per joint. "
            "Three model variants: 'rnea' (RNEA only), 'model' (RNEA+Friction), "
            "'fric' (Friction only). bin_edges has 101 values → 100 bins."
        ),
        "bin_edges":  HIST_BIN_EDGES,
        "global":     _ser(global_hists),
        "by_shape":   {sh: _ser(h) for sh, h in shape_hists.items()},
    }

    # ── per-run compact entries ───────────────────────────────
    gs["runs"] = []
    for s in summaries:
        entry = {
            "run_id":     s["run_id"],
            "label":      s["label"],
            "success":    s["success"],
            "N":          s["N"],
            "duration_s": s["duration_s"],
            "traj_meta":  s["traj_meta"],
        }
        for j in range(DOF):
            js = s["joints"][j]
            entry[f"j{j}_load_rms"]   = js.get("load_rms")
            entry[f"j{j}_resid_rms"]  = js.get("residual_rms")
            entry[f"j{j}_rnea_ratio"] = js.get("rnea_over_load")
            entry[f"j{j}_nrmse"]      = js.get("nrmse")
        gs["runs"].append(entry)

    gs["errors"] = errors
    return gs


# ─────────────────────────────────────────────────────────────
# Trajectory plot (one per run)
# ─────────────────────────────────────────────────────────────

def plot_trajectory(arrays: dict, save_path: str) -> None:
    """
    Plot Load vs RNEA vs RNEA+Friction for one trajectory, 6 subplots (2×3).

    Legend labels include per-joint RMS.  Figure title is the run_id.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_id     = arrays["run_id"]
    t          = arrays["t"]
    tau_load   = arrays["tau_load"]
    tau_rnea   = arrays["tau_rnea"]
    tau_model  = arrays["tau_model"]
    nq         = tau_load.shape[1]

    # Normalise time to seconds from start
    t_s = t - t[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
    axes = axes.flatten()

    for j in range(6):
        ax = axes[j]
        ax.set_title(f"Joint {j+1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Torque (N·m)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        if j >= nq:
            ax.text(0.5, 0.5, "passive / no model",
                    ha="center", va="center", transform=ax.transAxes,
                    color="grey", fontsize=9)
            ax.set_xlim(0, 1)
            continue

        load_rms  = float(np.sqrt(np.mean(tau_load[:, j] ** 2)))
        rnea_rms  = float(np.sqrt(np.mean(tau_rnea[:, j] ** 2)))
        model_rms = float(np.sqrt(np.mean(tau_model[:, j] ** 2)))

        ax.plot(t_s, tau_load[:, j],
                color="#2196F3", linewidth=0.9, alpha=0.85,
                label=f"Load  (RMS={load_rms:.3f} N·m)")
        ax.plot(t_s, tau_rnea[:, j],
                color="#F44336", linewidth=0.9, alpha=0.85,
                label=f"RNEA  (RMS={rnea_rms:.3f} N·m)")
        ax.plot(t_s, tau_model[:, j],
                color="#4CAF50", linewidth=0.9, alpha=0.85,
                label=f"RNEA+Fric (RMS={model_rms:.3f} N·m)")

        ax.legend(fontsize=7.5, loc="best", framealpha=0.7)

    fig.suptitle(f"Run: {run_id}", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Bulk torque data save
# ─────────────────────────────────────────────────────────────

def save_torque_data(all_arrays: list, output_dir: str) -> str:
    """
    Concatenate torque arrays from all runs and save as torque_data.npz.

    Keys in the .npz:
      run_ids       — string array, one entry per sample (for traceability)
      t             — (total_N,)     timestamps
      q             — (total_N, nq)  joint positions
      qd            — (total_N, nq)  joint velocities
      tau_load      — (total_N, nq)  load-register torque
      tau_rnea      — (total_N, nq)  RNEA torque
      tau_fric      — (total_N, nq)  friction torque
      tau_model     — (total_N, nq)  RNEA + friction
      tau_residual  — (total_N, nq)  tau_load − tau_model
    """
    if not all_arrays:
        return ""

    keys = ["t", "q", "qd", "tau_load", "tau_rnea", "tau_fric",
            "tau_model", "tau_residual"]

    # Build per-key lists and run_id label array
    concat = {k: [] for k in keys}
    run_id_list = []

    for arr in all_arrays:
        n = len(arr["t"])
        run_id_list.extend([arr["run_id"]] * n)
        for k in keys:
            concat[k].append(arr[k])

    save_dict = {k: np.concatenate(concat[k], axis=0) for k in keys}
    save_dict["run_ids"] = np.array(run_id_list, dtype=object)

    total_N = len(save_dict["t"])
    out_path = os.path.join(output_dir, "torque_data.npz")
    np.savez_compressed(out_path, **save_dict)
    logger.info("Saved torque_data.npz: %d samples from %d runs → %s",
                total_N, len(all_arrays), out_path)
    return out_path


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    C.setup_logging()
    set_headless(True)

    json_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not json_files:
        print(f"No JSON files found in {RAW_DIR}")
        return

    # --- Create run-specific output folder ---
    run_folder  = get_run_folder_name()
    OUTPUT_DIR  = os.path.join(BASE_OUT_DIR, run_folder)
    TRAJ_DIR    = os.path.join(OUTPUT_DIR, "trajectory_plots")
    GLOBAL_DIR  = os.path.join(OUTPUT_DIR, "global_plots")
    os.makedirs(TRAJ_DIR,   exist_ok=True)
    os.makedirs(GLOBAL_DIR, exist_ok=True)

    print(f"Found {len(json_files)} JSON files")
    print(f"Output: {OUTPUT_DIR}/")
    print("=" * 70)

    # --- Build Pinocchio model once ---
    print("Building Pinocchio model...")
    model, _, nq = build_pinocchio_model(
        C.XACRO_PATH, mass_scale=C.MASS_SCALE, extra_masses=C.EXTRA_MASSES,
    )
    total_mass = sum(model.inertias[i].mass for i in range(model.njoints))
    print(f"  nq={nq}  total_mass={total_mass:.4f} kg\n")

    # --- Per-file processing ---
    summaries  = []
    errors     = []
    all_arrays = []
    t_start    = time.time()

    for idx, jf in enumerate(json_files):
        fname = os.path.basename(jf)
        print(f"[{idx+1:3d}/{len(json_files)}] {fname}", end="  ")
        t0 = time.time()
        try:
            s, arr = process_one_file(jf, model)
            summaries.append(s)
            all_arrays.append(arr)

            # Trajectory plot — one per run
            safe_name = re.sub(r"[^\w\-.]", "_", arr["run_id"])
            plot_path = os.path.join(TRAJ_DIR, f"{safe_name}.png")
            plot_trajectory(arr, plot_path)

            dt = time.time() - t0
            r2 = s["joints"][1].get("rnea_over_load", "N/A")
            r3 = s["joints"][2].get("rnea_over_load", "N/A")
            print(f"OK {dt:.1f}s  J2={r2}  J3={r3}")
        except Exception as e:
            dt = time.time() - t0
            print(f"FAIL {dt:.1f}s  {e}")
            logger.error("Failed %s: %s\n%s", fname, e, traceback.format_exc())
            errors.append({"file": fname, "error": str(e),
                           "traceback": traceback.format_exc()})

    total_time = time.time() - t_start

    # --- Save concatenated torque data ---
    if all_arrays:
        print("\nSaving torque_data.npz ...")
        npz_path = save_torque_data(all_arrays, OUTPUT_DIR)
        print(f"  {npz_path}")

    # --- Global summary JSON ---
    gs       = build_global_summary(summaries, errors, total_time, model)
    gs_path  = os.path.join(OUTPUT_DIR, "global_summary.json")
    with open(gs_path, "w") as f:
        json.dump(gs, f, indent=2, default=str)

    # --- Global plots ---
    if summaries:
        print("\nGenerating global plots...")
        from Torque_Analysis.plots_global import generate_all_global_plots
        generate_all_global_plots(
            summaries,
            GLOBAL_DIR,
            error_histograms=gs.get("error_histograms"),
        )

    # --- Build report lines (shared between stdout and metadata.txt) ---
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"  Run folder: {run_folder}")
    report_lines.append(f"  Files:      {len(json_files)}")
    report_lines.append(f"  OK:         {len(summaries)}")
    report_lines.append(f"  Failed:     {len(errors)}")
    report_lines.append(f"  Time:       {total_time:.0f}s ({total_time/max(len(json_files),1):.1f}s avg)")
    report_lines.append(f"  Output:     {OUTPUT_DIR}/")
    report_lines.append(f"  Summary:    {gs_path}")

    if summaries:
        report_lines.append(f"  Samples:    {sum(s['N'] for s in summaries)}")
        report_lines.append("")
        report_lines.append("  Per-joint model quality (median RNEA/Load  |  median NRMSE):")
        for ja in gs.get("joint_aggregate", []):
            j      = ja["joint"]
            rr     = ja.get("rnea_over_load", {})
            nr     = ja.get("nrmse", {})
            med_r  = rr.get("median", "N/A")
            pct_g  = rr.get("pct_good", "?")
            med_nr = f"{(nr.get('median', 0) * 100):.1f}%" if nr else "N/A"
            note   = " (yaw)" if j == 0 else " (tool)" if j == 5 else ""
            report_lines.append(
                f"    J{j+1}{note:8s}: RNEA/Load={med_r}  good={pct_g}%  NRMSE={med_nr}"
            )

        # Shape-wise averages
        report_lines.append("")
        report_lines.append("  Shape-wise median NRMSE (active joints J1–J5):")
        header = f"    {'Shape':<18}" + "".join(f"  J{j+1:>6}" for j in range(5))
        report_lines.append(header)
        report_lines.append("    " + "-" * (18 + 5 * 8))
        for shape, sd in gs.get("by_shape", {}).items():
            vals = []
            for j in range(5):
                v = sd["joints"].get(f"j{j}", {}).get("nrmse_median")
                vals.append(f"{v*100:6.1f}%" if v is not None else f"{'N/A':>7}")
            report_lines.append(f"    {shape:<18}" + "  ".join(vals))

        # Trajectory-type averages
        report_lines.append("")
        report_lines.append("  Trajectory-type median NRMSE (active joints J1–J5):")
        report_lines.append(header.replace("Shape", "TrajType"))
        report_lines.append("    " + "-" * (18 + 5 * 8))
        for tt, td in gs.get("by_traj_type", {}).items():
            vals = []
            for j in range(5):
                v = td["joints"].get(f"j{j}", {}).get("nrmse_median")
                vals.append(f"{v*100:6.1f}%" if v is not None else f"{'N/A':>7}")
            report_lines.append(f"    {tt:<18}" + "  ".join(vals))

    if errors:
        report_lines.append("")
        report_lines.append(f"  Errors ({len(errors)}):")
        for e in errors[:5]:
            report_lines.append(f"    {e['file']}: {e['error']}")

    report_lines.append("=" * 70)

    # --- Print to stdout ---
    print("\n" + "\n".join(report_lines))

    # --- Write metadata.txt ---
    meta_path = os.path.join(OUTPUT_DIR, "metadata.txt")
    with open(meta_path, "w") as f:
        f.write("Torque Analysis — Bulk Run Report\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\n".join(report_lines) + "\n")
    print(f"\n  metadata.txt: {meta_path}")


if __name__ == "__main__":
    main()
