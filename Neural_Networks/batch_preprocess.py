#!/usr/bin/env python3
"""Headless batch preprocessing for filtering ablation study.

Builds 4 datasets from the 80 unused raw trajectories (those NOT already used
in run_train22_q0_qd91_qdd21_tau51_rnea15), applying different Savitzky-Golay
filtering configurations to find the best preprocessing for EDR training.

Critical hypothesis: run_train22 uses qdd_locked=False — qd (91-pt SG) and
qdd (21-pt SG) come from separate polynomial fits, breaking qdd = d/dt(qd).
This makes the RNEA inertia term M(q)·q̈ physically inconsistent.
Locking both to the same polynomial should reduce the RNEA residual and
give EDR a cleaner correction target.

Usage (from repository root):
    PYTHONPATH=. python3 Neural_Networks/batch_preprocess.py
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_NN_ROOT   = _REPO_ROOT / "Neural_Networks"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# xacro lives in the ROS Python path on this machine
_ROS_PY = "/opt/ros/jazzy/lib/python3.12/site-packages"
if Path(_ROS_PY).is_dir() and _ROS_PY not in sys.path:
    sys.path.insert(0, _ROS_PY)

from Neural_Networks.robot_physics import (  # noqa: E402
    ACTIVE_JOINTS,
    build_pinocchio_model,
    compute_rnea_decomposition,
    fix_timestamps,
    load_calibration_params,
    raw_derivatives,
    savgol_smooth,
    sg_differentiate,
    ticks_to_radians,
    torque_friction,
    torque_from_load,
)
from Neural_Networks.loader import JOINT_NAMES, load_raw_sample  # noqa: E402

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
RAW_SAMPLES_DIR = _REPO_ROOT / "raw_samples"
CALIB_PATH      = _REPO_ROOT / "Torque_Analysis" / "calibration_params.json"
XACRO_PATH      = _REPO_ROOT / "robot_description" / "urdf" / "kikobot.xacro"
REFERENCE_RUN   = _NN_ROOT / "train_data" / "run_train22_q0_qd91_qdd21_tau51_rnea15"
OUTPUT_BASE     = _NN_ROOT / "train_data"

RANDOM_SEED = 42
POLYORDER   = 3
SG_MODE     = "interp"
TRIM_PCT    = 1.0   # percent trimmed from each end


# ---------------------------------------------------------------------------
# Ablation configs
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    name: str
    qd_window: int
    qdd_window: int
    locked: bool    # True → same polynomial for qd & qdd (physically consistent)
    tau_window: int
    q_smooth: bool = False
    q_window: int  = 15


CONFIGS: list[FilterConfig] = [
    # Best config only (qd/qdd locked, 91-pt, physics-consistent)
    FilterConfig("q0_qd91_qdd91_tau51_lk_3to1",  qd_window=91, qdd_window=91, locked=True,  tau_window=51),
]


# ---------------------------------------------------------------------------
# Trajectory selection
# ---------------------------------------------------------------------------

def _used_source_files(reference_run: Path) -> set[str]:
    used: set[str] = set()
    for split in ("train", "val", "test"):
        cat = reference_run / split / "trajectories_catalog.csv"
        if cat.exists():
            with cat.open() as f:
                for row in csv.DictReader(f):
                    used.add(row["source_file"])
    return used


def _geometry_type(fname: str) -> str:
    m = re.match(r"^([a-z_]+?)_r\d", fname)
    return m.group(1) if m else "unknown"


def _planner_from_name(fname: str) -> str:
    for p in ("quintic_bezier", "quintic_poly", "cubic_poly", "trapezoidal", "ruckig"):
        if p in fname:
            return p.replace("_", " ").title().replace(" ", "")
    return "Unknown"


def build_stratified_split(
    raw_dir: Path,
    used: set[str],
    seed: int = RANDOM_SEED,
) -> dict[str, list[Path]]:
    """
    Returns {train, val, test} lists using 2:1:1 stratified split per geometry type.
    Excludes all filenames in `used`.
    """
    unused = sorted(p for p in raw_dir.glob("*.json") if p.name not in used)

    rng = random.Random(seed)
    groups: dict[str, list[Path]] = {}
    for p in unused:
        groups.setdefault(_geometry_type(p.name), []).append(p)

    train, val, test = [], [], []
    print("\n  Geometry-type breakdown (2:1:1 split):")
    print(f"  {'Type':<22} {'Total':>5}  {'Train':>5}  {'Val':>5}  {'Test':>5}")
    print("  " + "-" * 48)
    for gtype, files in sorted(groups.items()):
        shuffled = list(files)
        rng.shuffle(shuffled)
        n       = len(shuffled)
        n_train = (n * 3) // 5      # 3:1:1 → ~60% train
        n_val   = (n - n_train) // 2
        n_test  = n - n_train - n_val
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])
        print(f"  {gtype:<22} {n:>5}  {n_train:>5}  {n_val:>5}  {n_test:>5}")
    print("  " + "-" * 48)
    print(f"  {'TOTAL':<22} {len(unused):>5}  {len(train):>5}  {len(val):>5}  {len(test):>5}")
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Per-trajectory processing
# ---------------------------------------------------------------------------

def _apply_kinematics(
    q_filt: np.ndarray,
    dt: float,
    cfg: FilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (filtered_qd, filtered_qdd) using the config's locking strategy.
    locked=True  → both from same SG polynomial (qdd = d/dt(qd) by construction)
    locked=False → separate SG fits (replicates run_train22 baseline behaviour)
    """
    if cfg.locked:
        return sg_differentiate(q_filt, dt, cfg.qd_window, POLYORDER, SG_MODE)
    filt_qd, _ = sg_differentiate(q_filt, dt, cfg.qd_window,  POLYORDER, SG_MODE)
    _, filt_qdd = sg_differentiate(q_filt, dt, cfg.qdd_window, POLYORDER, SG_MODE)
    return filt_qd, filt_qdd


def process_trajectory(
    json_path: Path,
    cfg: FilterConfig,
    pin_model: Any,
    pin_data: Any,
    calib: dict,
) -> dict | None:
    try:
        L, M, _ = load_raw_sample(str(json_path))
    except Exception as e:
        print(f"\n    WARNING: failed to load {json_path.name}: {e}")
        return None

    N = len(L["t"])

    # Raw joint angles (N, 6) → take first ACTIVE_JOINTS
    q_full = ticks_to_radians(L["act_pos"], M["joint_map"], M["ticks_to_rad"], dof=6)
    q_raw  = q_full[:, :ACTIVE_JOINTS].astype(np.float32)

    # Raw torque (N, 6) → take first ACTIVE_JOINTS
    tau_full = torque_from_load(L["load"], L["voltage"], joint_map=M["joint_map"])
    tau_raw  = tau_full[:, :ACTIVE_JOINTS].astype(np.float32)

    # Timestamps
    tr = fix_timestamps(L["t"])
    t  = tr.t_fixed.astype(np.float64)
    dt = float(tr.median_dt)
    if dt <= 0:
        print(f"\n    WARNING: invalid dt ({dt}) in {json_path.name} — skipping")
        return None

    # Trim 1% front and back
    trim_n = max(1, int(N * TRIM_PCT / 100.0))
    sl = slice(trim_n, N - trim_n)
    t       = t[sl]
    q_raw   = q_raw[sl]
    tau_raw = tau_raw[sl]
    N_trim  = len(t)
    if N_trim < max(cfg.qd_window, cfg.qdd_window, cfg.tau_window) * 2:
        print(f"\n    WARNING: only {N_trim} samples after trim in {json_path.name} — skipping")
        return None

    # Raw derivatives (noisy, for raw_qd/raw_qdd reference columns only)
    raw_qd, raw_qdd = raw_derivatives(q_raw, t)

    # Filtered kinematics
    filt_q   = savgol_smooth(q_raw, cfg.q_window, POLYORDER) if cfg.q_smooth else q_raw.copy()
    filt_qd, filt_qdd = _apply_kinematics(filt_q, dt, cfg)

    # Filtered torque measurement
    filt_tau = savgol_smooth(tau_raw, cfg.tau_window, POLYORDER)

    # RNEA decomposition + friction
    coulomb = np.asarray(calib["coulomb_nm"][:ACTIVE_JOINTS], dtype=np.float64)
    viscous = np.asarray(calib["viscous_nm"][:ACTIVE_JOINTS], dtype=np.float64)
    eps     = float(calib["friction_eps"])

    tau_rnea, tau_g, tau_M, tau_C = compute_rnea_decomposition(
        pin_model, pin_data, filt_q, filt_qd, filt_qdd, n_active=ACTIVE_JOINTS
    )
    tau_f = torque_friction(filt_qd, coulomb=coulomb, viscous=viscous, eps=eps)

    # tau_decomposed layout: [τ_g(5), τ_M(5), τ_C(5), τ_f(5)]
    tau_decomposed = np.concatenate([tau_g, tau_M, tau_C, tau_f], axis=1).astype(np.float32)

    geo      = M.get("geometry") or {}
    tracking = M.get("tracking") or {}
    return {
        "t":                       t.astype(np.float32),
        "raw_q":                   q_raw,
        "raw_qd":                  raw_qd.astype(np.float32),
        "raw_qdd":                 raw_qdd.astype(np.float32),
        "raw_tau_measured":        tau_raw,
        "raw_tau_decomposed":      tau_decomposed,   # RNEA on filtered kinematics
        "filtered_q":              filt_q.astype(np.float32),
        "filtered_qd":             filt_qd.astype(np.float32),
        "filtered_qdd":            filt_qdd.astype(np.float32),
        "filtered_tau_measured":   filt_tau.astype(np.float32),
        "filtered_tau_decomposed": tau_decomposed,
        "source_file":             json_path.name,
        "ctrl_hz":                 float(M.get("ctrl_hz", 0.0)),
        "fb_hz":                   float(M.get("fb_hz", 0.0)),
        "duration_sec":            float(M.get("duration", 0.0)),
        "geometry":                geo,
        "tracking":                tracking,
    }


# ---------------------------------------------------------------------------
# Dataset saving
# ---------------------------------------------------------------------------

_JOINT_HDR  = ",".join(JOINT_NAMES[:ACTIVE_JOINTS])
_DECOMP_HDR = ",".join(
    [f"{j}_g" for j in JOINT_NAMES[:ACTIVE_JOINTS]]
    + [f"{j}_M" for j in JOINT_NAMES[:ACTIVE_JOINTS]]
    + [f"{j}_C" for j in JOINT_NAMES[:ACTIVE_JOINTS]]
    + [f"{j}_f" for j in JOINT_NAMES[:ACTIVE_JOINTS]]
)


def _save_csv(path: Path, arr: np.ndarray, header: str) -> None:
    np.savetxt(path, arr, delimiter=",", fmt="%.8f", header=header, comments="")


def save_split(split_dir: Path, records: list[dict]) -> int:
    split_dir.mkdir(parents=True, exist_ok=True)

    keys_j = ["raw_q", "raw_qd", "raw_qdd", "raw_tau_measured",
               "filtered_q", "filtered_qd", "filtered_qdd", "filtered_tau_measured"]
    keys_d = ["raw_tau_decomposed", "filtered_tau_decomposed"]

    cat_arrays: dict[str, list[np.ndarray]] = {k: [] for k in ["t"] + keys_j + keys_d}
    catalog_rows = []
    cursor = 0

    for r in records:
        n = len(r["t"])
        for k in ["t"] + keys_j + keys_d:
            cat_arrays[k].append(r[k])
        geo      = r.get("geometry") or {}
        tracking = r.get("tracking") or {}
        catalog_rows.append({
            "source_file":       r["source_file"],
            "geometry_type":     _geometry_type(r["source_file"]),
            "radius_mm":         geo.get("radius_mm", 0.0),
            "planner":           geo.get("planner") or _planner_from_name(r["source_file"]),
            "ctrl_hz":           r["ctrl_hz"],
            "fb_hz":             r["fb_hz"],
            "duration_sec":      r["duration_sec"],
            "n_samples":         n,
            "ee_rms_err_mm":     (tracking.get("ee_rms_err_mm") or 0.0),
            "start_idx":         cursor,
            "end_idx_exclusive": cursor + n,
        })
        cursor += n

    def _cat(k): return np.concatenate(cat_arrays[k], axis=0)

    t_arr = _cat("t")
    np.savetxt(split_dir / "t.csv", t_arr.reshape(-1, 1), delimiter=",",
               fmt="%.8f", header="t", comments="")

    for k in keys_j:
        fname = f"{k}.csv"
        _save_csv(split_dir / fname, _cat(k), _JOINT_HDR)
    for k in keys_d:
        fname = f"{k}.csv"
        _save_csv(split_dir / fname, _cat(k), _DECOMP_HDR)

    with (split_dir / "trajectories_catalog.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(catalog_rows[0].keys()))
        w.writeheader()
        w.writerows(catalog_rows)

    return cursor


def compute_norm_stats(train_dir: Path) -> dict:
    def _ld(name):
        return np.loadtxt(train_dir / name, delimiter=",", skiprows=1, dtype=np.float64)

    q    = _ld("filtered_q.csv")
    qd   = _ld("filtered_qd.csv")
    qdd  = _ld("filtered_qdd.csv")
    tau  = _ld("filtered_tau_measured.csv")
    tau_d = _ld("filtered_tau_decomposed.csv")
    tau_a = tau_d.reshape(-1, 4, ACTIVE_JOINTS).sum(axis=1)

    def _stats(arr):
        return arr.mean(axis=0).tolist(), np.clip(arr.std(axis=0), 1e-8, None).tolist()

    mq, sq     = _stats(q)
    mqd, sqd   = _stats(qd)
    mqdd, sqdd = _stats(qdd)
    mt, st     = _stats(tau)
    mta, sta   = _stats(tau_a)
    return {
        "mean_q": mq,    "std_q": sq,
        "mean_qd": mqd,  "std_qd": sqd,
        "mean_qdd": mqdd, "std_qdd": sqdd,
        "mean_tau": mt,  "std_tau": st,
        "mean_tau_a": mta, "std_tau_a": sta,
    }


def build_metadata(cfg: FilterConfig, run_dir: Path, processed: dict[str, list[dict]],
                   norm_stats: dict, timestamp: str) -> dict:
    def _sstat(records):
        traj_list = []
        cursor = 0
        for r in records:
            n = int(len(r["t"]))
            geo = r.get("geometry") or {}
            traj_list.append({
                "source_file":       r["source_file"],
                "geometry_type":     _geometry_type(r["source_file"]),
                "geometry":          geo,
                "tracking":          r.get("tracking") or {},
                "ctrl_hz":           r["ctrl_hz"],
                "fb_hz":             r["fb_hz"],
                "duration_sec":      r["duration_sec"],
                "n_samples":         n,
                "start_idx":         cursor,
                "end_idx_exclusive": cursor + n,
            })
            cursor += n
        return {
            "n_samples":      cursor,
            "n_trajectories": len(records),
            "trajectories":   traj_list,
        }

    return {
        "format":     "v4_ablation_metadata",
        "created_at": timestamp,
        "run_dir":    str(run_dir),
        "n_trajectories_total": sum(len(v) for v in processed.values()),
        "preprocessing": {
            "trim": {"front_percent": TRIM_PCT, "back_percent": TRIM_PCT},
            "q_smooth": {
                "enabled": cfg.q_smooth,
                "method":  "savgol" if cfg.q_smooth else "none",
                "window_length": cfg.q_window if cfg.q_smooth else None,
                "polyorder":     POLYORDER if cfg.q_smooth else None,
            },
            "differentiation": {
                "method": "savgol",
                "qd": {
                    "window_length": cfg.qd_window,
                    "polyorder":     POLYORDER,
                    "mode":          SG_MODE,
                    "note":          "deriv=1 of Savitzky-Golay on filtered_q",
                },
                "qdd": {
                    "locked_to_qd":  cfg.locked,
                    "window_length": cfg.qd_window if cfg.locked else cfg.qdd_window,
                    "polyorder":     POLYORDER,
                    "mode":          SG_MODE,
                    "note": ("deriv=2 from same SG fit (locked)" if cfg.locked
                             else "deriv=2 from separate SG fit (not locked)"),
                },
            },
            "tau_measured_smooth": {
                "enabled":       True,
                "method":        "savgol",
                "window_length": cfg.tau_window,
                "polyorder":     POLYORDER,
            },
            "tau_analytical": {
                "rnea_enabled":  True,
                "friction_model": "coulomb_viscous",
                "source":        "RNEA(filtered_q, filtered_qd, filtered_qdd) + friction(filtered_qd)",
                "output_format": "decomposed_20dim",
                "layout":        "tau_g(5), tau_M(5), tau_C(5), tau_f(5)",
            },
            "tau_analytical_postfilter": {"enabled": False, "method": "none"},
        },
        "split": {
            "mode":     "stratified",
            "strategy": "stratified_by_geometry_type_2_1_1",
            "ratios":   {"train": 0.5, "val": 0.25, "test": 0.25},
            "stats": {
                "train": _sstat(processed["train"]),
                "val":   _sstat(processed["val"]),
                "test":  _sstat(processed["test"]),
            },
        },
        "normalisation": norm_stats,
    }


# ---------------------------------------------------------------------------
# Per-config builder
# ---------------------------------------------------------------------------

def build_dataset(
    cfg: FilterConfig,
    split_files: dict[str, list[Path]],
    pin_model: Any,
    pin_data: Any,
    calib: dict,
) -> tuple[Path, dict[str, list[dict]]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name  = f"run_abl_{cfg.name}_{timestamp}"
    run_dir   = OUTPUT_BASE / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  Config: {cfg.name}")
    print(f"  Output: {run_name}")
    print(f"{'='*72}")

    processed: dict[str, list[dict]] = {}
    for split_name, paths in split_files.items():
        print(f"\n  [{split_name.upper()}] {len(paths)} trajectories")
        records = []
        for i, p in enumerate(paths, 1):
            print(f"    [{i:2d}/{len(paths)}] {p.name[:60]}", end=" ", flush=True)
            result = process_trajectory(p, cfg, pin_model, pin_data, calib)
            if result is not None:
                records.append(result)
                print(f"→ {len(result['t']):,} samples")
            else:
                print("→ SKIPPED")
        if not records:
            raise RuntimeError(f"No valid trajectories in {split_name}")
        processed[split_name] = records

    # Write CSVs
    for split_name, records in processed.items():
        total = save_split(run_dir / split_name, records)
        n_traj = len(records)
        print(f"  Saved {split_name}: {n_traj} trajectories, {total:,} samples")

    # Normalization from train
    print("  Computing normalization stats...", end=" ", flush=True)
    norm_stats = compute_norm_stats(run_dir / "train")
    print("done.")

    # metadata.json
    meta = build_metadata(cfg, run_dir, processed, norm_stats, datetime.now().isoformat())
    with (run_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    return run_dir, processed


# ---------------------------------------------------------------------------
# RNEA residual analysis
# ---------------------------------------------------------------------------

def rnea_residual(run_dir: Path) -> dict:
    train_dir  = run_dir / "train"
    tau_meas   = np.loadtxt(train_dir / "filtered_tau_measured.csv",
                            delimiter=",", skiprows=1, dtype=np.float64)
    tau_decomp = np.loadtxt(train_dir / "filtered_tau_decomposed.csv",
                            delimiter=",", skiprows=1, dtype=np.float64)
    tau_ana    = tau_decomp.reshape(-1, 4, ACTIVE_JOINTS).sum(axis=1)  # (N, 5)
    residual   = tau_meas - tau_ana
    return {
        "rmse_pooled":    float(np.sqrt((residual ** 2).mean())),
        "rmse_per_joint": np.sqrt((residual ** 2).mean(axis=0)).tolist(),
        "mae_pooled":     float(np.abs(residual).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("  Filtering Ablation Dataset Builder")
    print("=" * 72)
    print(f"  Raw samples : {RAW_SAMPLES_DIR}")
    print(f"  Reference   : {REFERENCE_RUN.name}")
    print(f"  Output base : {OUTPUT_BASE}")

    # Load calibration
    calib = load_calibration_params(str(CALIB_PATH))
    print(f"\n  Calibration: coulomb={calib['coulomb_nm'][:ACTIVE_JOINTS]}")
    print(f"               viscous={calib['viscous_nm'][:ACTIVE_JOINTS]}")

    # Build Pinocchio model (once, shared across all configs)
    print("\n  Loading Pinocchio model...", end=" ", flush=True)
    pin_model, pin_data, _ = build_pinocchio_model(str(XACRO_PATH))
    print("done.")

    # Determine unused trajectories
    print(f"\n  Loading exclusion list from reference run...", end=" ", flush=True)
    used = _used_source_files(REFERENCE_RUN)
    print(f"{len(used)} files excluded.")

    # Build stratified split (same for all 4 configs)
    split_files = build_stratified_split(RAW_SAMPLES_DIR, used)
    n_total = sum(len(v) for v in split_files.values())
    print(f"\n  Total unused trajectories selected: {n_total}")

    # Verify zero overlap
    all_sel = {p.name for ps in split_files.values() for p in ps}
    overlap = all_sel & used
    if overlap:
        raise RuntimeError(f"BUG: overlap with reference run: {overlap}")

    # Build all 4 datasets
    results: dict[str, dict] = {}
    for cfg in CONFIGS:
        run_dir, _ = build_dataset(cfg, split_files, pin_model, pin_data, calib)
        residuals  = rnea_residual(run_dir)
        results[cfg.name] = {"run_dir": run_dir, "residuals": residuals}

    # Final summary table
    print(f"\n\n{'='*92}")
    print("  RNEA RESIDUAL SUMMARY  (τ_measured − τ_analytical, train split)")
    print("  Smaller residual → smaller EDR correction target → easier to learn")
    print(f"{'='*92}")
    hdr = f"  {'Config':<32} {'Pooled RMSE':>12} {'MAE':>10}  {'J1':>8} {'J2':>8} {'J3':>8} {'J4':>8} {'J5':>8}"
    print(hdr)
    print("  " + "-" * 88)
    best_name, best_rmse = "", float("inf")
    for name, r in results.items():
        res = r["residuals"]
        pj  = res["rmse_per_joint"]
        print(f"  {name:<32} {res['rmse_pooled']:>12.5f} {res['mae_pooled']:>10.5f}"
              + "  " + " ".join(f"{v:>8.5f}" for v in pj))
        if res["rmse_pooled"] < best_rmse:
            best_rmse = res["rmse_pooled"]
            best_name = name
    print(f"{'='*92}")
    print(f"\n  Best config: {best_name}  (RNEA residual RMSE = {best_rmse:.5f} N·m)")
    print(f"  → Recommended for EDR training.\n")
    print("  Dataset directories created:")
    for name, r in results.items():
        marker = " ← best" if name == best_name else ""
        print(f"    {r['run_dir'].name}{marker}")


if __name__ == "__main__":
    main()
