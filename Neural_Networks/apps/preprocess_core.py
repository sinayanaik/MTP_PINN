"""Headless dataset build (_do_build) — no GUI."""

from __future__ import annotations

import collections
import csv
import json as _json
import os
import re as _re
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NN_DIR = _HERE.parent.parent
_PROJECT_ROOT = _NN_DIR.parent

from Neural_Networks.apps.loader import (
    load_raw_sample,
    resolve_front_back_trim,
    CSV_T,
    CSV_RAW_Q,
    CSV_RAW_QD,
    CSV_RAW_QDD,
    CSV_RAW_TAU_MEASURED,
    CSV_RAW_TAU_DECOMPOSED,
    CSV_FILTERED_Q,
    CSV_FILTERED_QD,
    CSV_FILTERED_QDD,
    CSV_FILTERED_TAU_MEASURED,
    CSV_FILTERED_TAU_DECOMPOSED,
    METADATA_FILE,
    JOINT_NAMES,
)
from Neural_Networks.physics import (
    ACTIVE_JOINTS,
    fix_timestamps,
    ticks_to_radians,
    torque_from_load,
    torque_friction,
    savgol_smooth,
    sg_differentiate,
    raw_derivatives,
    build_pinocchio_model,
    compute_rnea_decomposition,
)

XACRO_PATH = _PROJECT_ROOT / "robot_description" / "urdf" / "kikobot.xacro"


def filtered_qd_qdd_from_params(
    filt_q: np.ndarray,
    dt: float,
    qd_w: int,
    qd_p: int,
    qd_mode: str,
    qdd_locked: bool,
    qdd_w: int,
    qdd_p: int,
    qdd_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filtered joint velocity and acceleration from filtered_q.

    When qdd_locked, qd and qdd are from one SG fit (qd params). When unlocked,
    qd uses the qd fit and qdd uses a separate SG fit (qdd params) for deriv=2.
    """
    if qdd_locked:
        return sg_differentiate(filt_q, dt, qd_w, qd_p, qd_mode)
    filt_qd, _ = sg_differentiate(filt_q, dt, qd_w, qd_p, qd_mode)
    _, filt_qdd = sg_differentiate(filt_q, dt, qdd_w, qdd_p, qdd_mode)
    return filt_qd, filt_qdd


def _do_build(params: dict) -> dict:
    """Process all trajectory files and save 10 CSVs + t.csv per split."""
    raw_dir   = params["raw_dir"]
    run_dir   = params["run_dir"]
    tr_ratio  = params["train_ratio"]
    vl_ratio  = params["val_ratio"]
    te_ratio  = params["test_ratio"]
    trim_fp   = params["trim_front_pct"]
    trim_bp   = params["trim_back_pct"]

    q_smooth  = params["q_smooth_enabled"]
    q_win     = params["q_window"]
    q_poly    = params["q_polyorder"]
    d_win     = params["deriv_window"]
    d_poly    = params["deriv_polyorder"]
    d_mode    = params["deriv_mode"]
    tau_smooth = params["tau_smooth_enabled"]
    tau_win   = params["tau_window"]
    tau_poly  = params["tau_polyorder"]
    use_rnea  = params["use_rnea"]
    pf_on     = params["tau_ana_postfilter_enabled"]
    pf_win    = params["tau_ana_window"]
    pf_poly   = params["tau_ana_polyorder"]
    qdd_locked = params.get("qdd_locked", True)
    qdd_w = params.get("qdd_window", d_win)
    qdd_p_ord = params.get("qdd_polyorder", d_poly)
    qdd_mode = params.get("qdd_mode", d_mode)

    log_lines = []
    def _log(msg):
        log_lines.append(msg)

    # Collect JSON files
    json_files = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {raw_dir}")
    _log(f"Found {len(json_files)} trajectory files")

    # ── Geometry composition filtering ──────────────────────────────────────
    # When geom_config is provided, select only the requested count per type
    # and respect the include/exclude flags.
    import re as _re

    def _geom_from_stem(stem: str) -> str:
        m = _re.match(r'^([a-z_]+?)_r\d', stem)
        return m.group(1) if m else "unknown"

    geom_config: dict = params.get("geom_config", {})
    if geom_config:
        from collections import defaultdict as _dd
        grps_by_type: dict = _dd(list)
        for jf in json_files:
            grps_by_type[_geom_from_stem(Path(jf).stem)].append(jf)

        rng_sel = np.random.default_rng(42)
        filtered_files: list = []
        _log("Geometry selection:")
        _log(f"  {'Type':<22} {'Avail':>6}  {'Used':>5}  Assignment")
        for gt in sorted(grps_by_type):
            files_for_gt = grps_by_type[gt]
            cfg = geom_config.get(gt, {})
            if not cfg.get("include", True):
                _log(f"  {gt:<22} {len(files_for_gt):>6}  {'--':>5}  EXCLUDED")
                continue
            assign = cfg.get("assignment", "all splits (stratified)")
            if assign == "exclude":
                _log(f"  {gt:<22} {len(files_for_gt):>6}  {'--':>5}  EXCLUDED")
                continue
            n_sel = min(cfg.get("count", len(files_for_gt)), len(files_for_gt))
            n_sel = max(1, n_sel)
            shuffled = list(files_for_gt)
            rng_sel.shuffle(shuffled)
            selected = shuffled[:n_sel]
            filtered_files.extend(selected)
            _log(f"  {gt:<22} {len(files_for_gt):>6}  {n_sel:>5}  {assign}")
        json_files = sorted(filtered_files)
        if not json_files:
            raise RuntimeError("No trajectories remain after geometry filtering. "
                               "Check your composition settings.")
        _log(f"Selected {len(json_files)} files after composition filtering")
    else:
        _log("No geometry composition config — using all files")


    # Build pinocchio model
    pin_model = pin_data = None
    if use_rnea:
        xacro = str(XACRO_PATH) if XACRO_PATH.exists() else None
        try:
            pin_model, pin_data, _ = build_pinocchio_model(xacro)
            _log("Pinocchio model loaded")
        except Exception as e:
            _log(f"Pinocchio error: {e} -- RNEA disabled")
            use_rnea = False

    nj = ACTIVE_JOINTS

    # Process each trajectory
    all_trajectories = []
    for i, jpath in enumerate(json_files):
        fname = os.path.basename(jpath)
        try:
            L, M, _ = load_raw_sample(jpath)
        except Exception as e:
            _log(f"  SKIP {fname}: {e}")
            continue

        # Ticks -> rad, fix timestamps, torque from load
        q_full = ticks_to_radians(L["act_pos"], M["joint_map"], M["ticks_to_rad"], M["dof"])
        ts_report = fix_timestamps(L["t"])
        t = ts_report.t_fixed.astype(np.float32)
        tau_full = torque_from_load(L["load"], L["voltage"], M["joint_map"])

        # Slice to active joints
        raw_q = q_full[:, :nj].astype(np.float32)
        raw_tau_m = tau_full[:, :nj].astype(np.float32)

        # Trim
        n_full = len(t)
        try:
            fn, bn = resolve_front_back_trim(n_full, trim_fp, trim_bp)
        except ValueError:
            fn, bn = 0, 0
        end = n_full - bn if bn > 0 else n_full
        sl = slice(fn, end)
        raw_q     = raw_q[sl]
        raw_tau_m = raw_tau_m[sl]
        t         = t[sl]

        dt = ts_report.median_dt
        if dt <= 0:
            dt = 1.0 / 300.0

        # Raw derivatives (np.gradient)
        raw_qd, raw_qdd = raw_derivatives(raw_q, t)

        # Raw tau_analytical
        if use_rnea and pin_model is not None:
            r_rnea, r_tau_g, r_tau_M, r_tau_C = compute_rnea_decomposition(
                pin_model, pin_data, raw_q, raw_qd, raw_qdd, n_active=nj)
            r_fric = torque_friction(raw_qd[:, :nj])
            raw_tau_decomp = np.concatenate([r_tau_g, r_tau_M, r_tau_C, r_fric], axis=1)  # (N, 20)
        else:
            raw_tau_decomp = np.zeros((raw_q.shape[0], 4 * nj), dtype=np.float64)

        # Filtered q
        filt_q = savgol_smooth(raw_q, q_win, q_poly) if q_smooth else raw_q.copy()

        # Filtered qd, qdd (same rules as GUI: lock uses one SG fit; unlock uses qd params for qd, qdd for qdd)
        filt_qd, filt_qdd = filtered_qd_qdd_from_params(
            filt_q, dt, d_win, d_poly, d_mode, qdd_locked, qdd_w, qdd_p_ord, qdd_mode)

        # Filtered tau_measured
        filt_tau_m = savgol_smooth(raw_tau_m, tau_win, tau_poly) if tau_smooth else raw_tau_m.copy()

        # Filtered tau_analytical (RNEA from filtered kinematics + friction)
        if use_rnea and pin_model is not None:
            f_rnea, f_tau_g, f_tau_M, f_tau_C = compute_rnea_decomposition(
                pin_model, pin_data, filt_q, filt_qd, filt_qdd, n_active=nj)
            f_fric = torque_friction(filt_qd[:, :nj])
            if pf_on:
                f_tau_g = savgol_smooth(f_tau_g, pf_win, pf_poly)
                f_tau_M = savgol_smooth(f_tau_M, pf_win, pf_poly)
                f_tau_C = savgol_smooth(f_tau_C, pf_win, pf_poly)
                f_fric  = savgol_smooth(f_fric,  pf_win, pf_poly)
            filt_tau_decomp = np.concatenate([f_tau_g, f_tau_M, f_tau_C, f_fric], axis=1)  # (N, 20)
            filt_tau_a = f_rnea + torque_friction(filt_qd[:, :nj])  # 5-dim total for reference only
        else:
            filt_tau_decomp = np.zeros((filt_q.shape[0], 4 * nj), dtype=np.float64)
            filt_tau_a = np.zeros_like(filt_q)

        geom = M.get("geometry", {})
        if isinstance(geom, dict):
            geom_type = geom.get("type", "unknown")
            geom_full = {
                "type":          geom.get("type", "unknown"),
                "radius_mm":     geom.get("actual_radius_mm", geom.get("radius_mm", None)),
                "center_m":      geom.get("center_m", None),
                "normal":        geom.get("normal", None),
                "planner":       geom.get("planner", None),
                "num_waypoints": geom.get("num_waypoints", None),
                "instance":      geom.get("instance", None),
            }
        else:
            geom_type = str(geom)
            geom_full = {"type": geom_type}

        tracking = M.get("tracking", {})
        if not isinstance(tracking, dict):
            tracking = {}

        all_trajectories.append({
            "t": t,
            "raw_q": raw_q, "raw_qd": raw_qd, "raw_qdd": raw_qdd,
            "raw_tau_measured": raw_tau_m, "raw_tau_decomposed": raw_tau_decomp,
            "filtered_q": filt_q, "filtered_qd": filt_qd, "filtered_qdd": filt_qdd,
            "filtered_tau_measured": filt_tau_m, "filtered_tau_decomposed": filt_tau_decomp,
            "source_file": M.get("source_file", fname),
            "geometry_type": geom_type,
            "geometry_full": geom_full,
            "tracking":      tracking,
            "ctrl_hz":       M.get("ctrl_hz", None),
            "fb_hz":         M.get("fb_hz", None),
            "duration":      float(M.get("duration", 0)),
            "original_samples": n_full,
            "trim_front": fn, "trim_back": bn,
            "ts_report": ts_report.to_dict(),
        })
        if (i + 1) % 20 == 0:
            _log(f"  Processed {i+1}/{len(json_files)}")

    if not all_trajectories:
        raise RuntimeError("No trajectories could be processed.")
    _log(f"Processed {len(all_trajectories)} trajectories")

    # Split: stratified (default), random, or temporal
    split_mode = str(params.get("split_mode", "stratified")).strip().lower()
    if split_mode == "temporal":
        indices = np.arange(len(all_trajectories), dtype=np.int64)
        n_train = int(len(indices) * tr_ratio)
        n_val   = int(len(indices) * vl_ratio)
        splits = {
            "train": list(indices[:n_train]),
            "val":   list(indices[n_train:n_train + n_val]),
            "test":  list(indices[n_train + n_val:]),
        }
        _log("Split mode: temporal (trajectories in sorted filename order)")
    elif split_mode == "random":
        rng_r = np.random.default_rng(42)
        indices = rng_r.permutation(len(all_trajectories))
        n_train = int(len(indices) * tr_ratio)
        n_val   = int(len(indices) * vl_ratio)
        splits = {
            "train": list(indices[:n_train]),
            "val":   list(indices[n_train:n_train + n_val]),
            "test":  list(indices[n_train + n_val:]),
        }
        _log("Split mode: random (seed=42)")
    else:  # "stratified" (default) or assignment-aware
        # Separate pinned trajectories from the stratified pool
        _geom_cfg = params.get("geom_config", {})
        pinned_train: list = []
        pinned_val:   list = []
        pinned_test:  list = []
        stratified_indices: list = []

        groups: dict = collections.defaultdict(list)
        for idx, traj in enumerate(all_trajectories):
            groups[traj["geometry_type"]].append(idx)

        rng_s = np.random.default_rng(42)
        _log("Split mode: stratified by geometry type (with assignment pins, seed=42)")
        _log(f"  {'Geometry':<26} {'Total':>5}  {'Pin':<10}  {'Train':>5}  {'Val':>5}  {'Test':>5}")

        for gtype in sorted(groups):
            grp = list(groups[gtype])
            rng_s.shuffle(grp)
            n = len(grp)
            assign = _geom_cfg.get(gtype, {}).get("assignment", "all splits (stratified)")

            if assign == "train only":
                pinned_train.extend(grp)
                _log(f"  {gtype:<26} {n:>5}  {'train':10}  {n:>5}  {'0':>5}  {'0':>5}")
            elif assign == "val only":
                pinned_val.extend(grp)
                _log(f"  {gtype:<26} {n:>5}  {'val':10}  {'0':>5}  {n:>5}  {'0':>5}")
            elif assign == "test only":
                pinned_test.extend(grp)
                _log(f"  {gtype:<26} {n:>5}  {'test':10}  {'0':>5}  {'0':>5}  {n:>5}")
            else:  # "all splits (stratified)" — proportional
                stratified_indices.extend(grp)
                # Per-geometry proportional allocation logged below

        # Run proportional allocation on the stratified pool
        train_idx: list = []
        val_idx:   list = []
        test_idx:  list = []

        if stratified_indices:
            strat_groups: dict = collections.defaultdict(list)
            for idx in stratified_indices:
                strat_groups[all_trajectories[idx]["geometry_type"]].append(idx)

            rng_s2 = np.random.default_rng(42)
            for gtype in sorted(strat_groups):
                grp = strat_groups[gtype]
                rng_s2.shuffle(grp)
                n = len(grp)
                n_tr = max(1, round(n * tr_ratio))
                remaining = n - n_tr
                if remaining > 0 and (vl_ratio + te_ratio) > 0:
                    n_vl = round(remaining * vl_ratio / (vl_ratio + te_ratio))
                    n_vl = min(max(n_vl, 0), remaining)
                else:
                    n_vl = 0
                if n >= 2 and (n_vl + (n - n_tr - n_vl)) == 0:
                    n_tr = n - 1
                    n_vl = 1
                n_te = n - n_tr - n_vl
                train_idx.extend(grp[:n_tr])
                val_idx.extend(grp[n_tr:n_tr + n_vl])
                test_idx.extend(grp[n_tr + n_vl:])
                _log(f"  {gtype:<26} {n:>5}  {'stratified':10}  {n_tr:>5}  {n_vl:>5}  {n_te:>5}")

        # Merge pinned + stratified
        final_train = pinned_train + train_idx
        final_val   = pinned_val   + val_idx
        final_test  = pinned_test  + test_idx

        rng_s3 = np.random.default_rng(42)
        rng_s3.shuffle(final_train)
        rng_s3.shuffle(final_val)
        rng_s3.shuffle(final_test)

        if not final_train:
            _log("  WARNING: train split is empty — check assignment pins")
        if not final_val:
            _log("  WARNING: val split is empty — check assignment pins")
        if not final_test:
            _log("  WARNING: test split is empty — check assignment pins")

        splits = {"train": final_train, "val": final_val, "test": final_test}


    header = ",".join(JOINT_NAMES)
    header_decomp = ",".join(
        [f"{j}_g" for j in JOINT_NAMES] +
        [f"{j}_M" for j in JOINT_NAMES] +
        [f"{j}_C" for j in JOINT_NAMES] +
        [f"{j}_f" for j in JOINT_NAMES]
    )
    os.makedirs(run_dir, exist_ok=True)

    split_stats = {}
    all_norm_data = {"q": [], "qd": [], "qdd": [], "tau": [], "tau_a": []}

    for split_name, idx_arr in splits.items():
        split_dir = os.path.join(run_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Concatenate
        arrays = {k: [] for k in [
            "t", "raw_q", "raw_qd", "raw_qdd", "raw_tau_measured", "raw_tau_decomposed",
            "filtered_q", "filtered_qd", "filtered_qdd",
            "filtered_tau_measured", "filtered_tau_decomposed"]}
        traj_entries = []
        offset = 0

        for idx in idx_arr:
            traj = all_trajectories[idx]
            n_s = len(traj["t"])
            for k in arrays:
                arrays[k].append(traj[k])
            traj_entries.append({
                "source_file":       traj["source_file"],
                "geometry_type":     traj["geometry_type"],
                "geometry":          traj["geometry_full"],
                "tracking":          traj["tracking"],
                "ctrl_hz":           traj["ctrl_hz"],
                "fb_hz":             traj["fb_hz"],
                "duration_sec":      traj["duration"],
                "original_samples":  traj["original_samples"],
                "trim_front":        traj["trim_front"],
                "trim_back":         traj["trim_back"],
                "n_samples":         n_s,
                "start_idx":         offset,
                "end_idx_exclusive": offset + n_s,
                "timestamp_repair":  traj["ts_report"],
            })
            offset += n_s

        for k in arrays:
            arrays[k] = np.concatenate(arrays[k], axis=0)

        # Save CSVs with headers
        def _save(fname, arr, hdr=None):
            path = os.path.join(split_dir, fname)
            if arr.ndim == 1:
                np.savetxt(path, arr.reshape(-1, 1), delimiter=",", fmt="%.8f",
                           header="t", comments="")
            else:
                np.savetxt(path, arr, delimiter=",", fmt="%.8f",
                           header=hdr or header, comments="")

        _save(CSV_T, arrays["t"])
        _save(CSV_RAW_Q, arrays["raw_q"])
        _save(CSV_RAW_QD, arrays["raw_qd"])
        _save(CSV_RAW_QDD, arrays["raw_qdd"])
        _save(CSV_RAW_TAU_MEASURED, arrays["raw_tau_measured"])
        _save(CSV_RAW_TAU_DECOMPOSED, arrays["raw_tau_decomposed"], header_decomp)
        _save(CSV_FILTERED_Q, arrays["filtered_q"])
        _save(CSV_FILTERED_QD, arrays["filtered_qd"])
        _save(CSV_FILTERED_QDD, arrays["filtered_qdd"])
        _save(CSV_FILTERED_TAU_MEASURED, arrays["filtered_tau_measured"])
        _save(CSV_FILTERED_TAU_DECOMPOSED, arrays["filtered_tau_decomposed"], header_decomp)

        n_total = arrays["filtered_q"].shape[0]
        split_stats[split_name] = {
            "n_samples": n_total,
            "n_trajectories": len(idx_arr),
            "trajectories": traj_entries,
        }

        if split_name == "train":
            all_norm_data["q"].append(arrays["filtered_q"])
            all_norm_data["qd"].append(arrays["filtered_qd"])
            all_norm_data["qdd"].append(arrays["filtered_qdd"])
            all_norm_data["tau"].append(arrays["filtered_tau_measured"])
            # Compute tau_a norm from RNEA total (sum of all 4 component blocks) — same joint space.
            decomp = arrays["filtered_tau_decomposed"]  # (N, 20)
            nj = ACTIVE_JOINTS
            tau_a_total = (decomp[:, :nj] + decomp[:, nj:2*nj]
                           + decomp[:, 2*nj:3*nj] + decomp[:, 3*nj:])  # (N, 5)
            all_norm_data["tau_a"].append(tau_a_total)

        _log(f"  {split_name.upper()}: {len(idx_arr)} traj, {n_total:,} samples")

        # Write trajectories_catalog.csv for this split
        catalog_fields = [
            "source_file", "geometry_type", "radius_mm", "planner",
            "ctrl_hz", "fb_hz", "duration_sec", "n_samples",
            "ee_rms_err_mm", "start_idx", "end_idx_exclusive",
        ]
        catalog_path = os.path.join(split_dir, "trajectories_catalog.csv")
        with open(catalog_path, "w", newline="") as _cf:
            writer = csv.DictWriter(_cf, fieldnames=catalog_fields, extrasaction="ignore")
            writer.writeheader()
            for te in traj_entries:
                geom_d  = te["geometry"] if isinstance(te["geometry"], dict) else {}
                track_d = te["tracking"] if isinstance(te["tracking"], dict) else {}
                writer.writerow({
                    "source_file":       te["source_file"],
                    "geometry_type":     te["geometry_type"],
                    "radius_mm":         geom_d.get("radius_mm", ""),
                    "planner":           geom_d.get("planner", ""),
                    "ctrl_hz":           te.get("ctrl_hz", ""),
                    "fb_hz":             te.get("fb_hz", ""),
                    "duration_sec":      te.get("duration_sec", ""),
                    "n_samples":         te["n_samples"],
                    "ee_rms_err_mm":     track_d.get("ee_rms_err_mm", ""),
                    "start_idx":         te["start_idx"],
                    "end_idx_exclusive": te["end_idx_exclusive"],
                })

    # Normalisation from training split
    q_all    = np.concatenate(all_norm_data["q"])
    qd_all   = np.concatenate(all_norm_data["qd"])
    qdd_all  = np.concatenate(all_norm_data["qdd"])
    tau_all  = np.concatenate(all_norm_data["tau"])
    tau_a_all = np.concatenate(all_norm_data["tau_a"])

    normalisation = {
        "mean_q":     q_all.mean(axis=0).tolist(),
        "std_q":      q_all.std(axis=0).clip(min=1e-8).tolist(),
        "mean_qd":    qd_all.mean(axis=0).tolist(),
        "std_qd":     qd_all.std(axis=0).clip(min=1e-8).tolist(),
        "mean_qdd":   qdd_all.mean(axis=0).tolist(),
        "std_qdd":    qdd_all.std(axis=0).clip(min=1e-8).tolist(),
        "mean_tau":   tau_all.mean(axis=0).tolist(),
        "std_tau":    tau_all.std(axis=0).clip(min=1e-8).tolist(),
        "mean_tau_a": tau_a_all.mean(axis=0).tolist(),
        "std_tau_a":  tau_a_all.std(axis=0).clip(min=1e-8).tolist(),
    }

    # Geometry distribution across splits
    geom_dist: dict = collections.defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for _sn, _idx_arr in splits.items():
        for _idx in _idx_arr:
            geom_dist[all_trajectories[_idx]["geometry_type"]][_sn] += 1
    geom_dist = {k: dict(v) for k, v in sorted(geom_dist.items())}

    # metadata.json
    meta_doc = {
        "format": "v4_stratified_metadata",
        "created_at": datetime.now().isoformat(),
        "raw_dir": raw_dir,
        "run_dir": run_dir,
        "n_trajectories_total": len(json_files),
        "n_trajectories_processed": len(all_trajectories),

        "preprocessing": {
            "trim": {"front_percent": trim_fp, "back_percent": trim_bp},
            "q_smooth": {
                "enabled": q_smooth,
                "method": "savgol" if q_smooth else "none",
                "window_length": q_win,
                "polyorder": q_poly,
            },
            "differentiation": {
                "method": "savgol",
                "qd": {
                    "window_length": d_win,
                    "polyorder": d_poly,
                    "mode": d_mode,
                    "note": "deriv=1 of Savitzky-Golay on filtered_q",
                },
                "qdd": {
                    "locked_to_qd": qdd_locked,
                    "window_length": d_win if qdd_locked else qdd_w,
                    "polyorder": d_poly if qdd_locked else qdd_p_ord,
                    "mode": d_mode if qdd_locked else qdd_mode,
                    "note": (
                        "deriv=2 from same SG fit as qd when locked; "
                        "separate SG fit for deriv=2 when unlocked"
                    ),
                },
            },
            "tau_measured_smooth": {
                "enabled": tau_smooth,
                "method": "savgol" if tau_smooth else "none",
                "window_length": tau_win,
                "polyorder": tau_poly,
            },
            "tau_analytical": {
                "rnea_enabled": use_rnea,
                "xacro_path": str(XACRO_PATH) if use_rnea else None,
                "friction_model": "coulomb_viscous",
                "source": "RNEA(filtered_q, filtered_qd, filtered_qdd) + friction(filtered_qd)",
                "output_format": "decomposed_20dim",
                "layout": "tau_g(5), tau_M(5), tau_C(5), tau_f(5)",
            },
            "tau_analytical_postfilter": {
                "enabled": pf_on,
                "method": "savgol" if pf_on else "none",
                "window_length": pf_win,
                "polyorder": pf_poly,
            },
        },

        "split": {
            "mode": split_mode,
            "strategy": (
                "stratified_by_geometry_type" if split_mode == "stratified"
                else split_mode
            ),
            "geometry_distribution": geom_dist,
            "geometry_config": geom_config if geom_config else None,
            "ratios": {"train": tr_ratio, "val": vl_ratio, "test": te_ratio},
            "stats": {k: {
                "n_samples":      v["n_samples"],
                "n_trajectories": v["n_trajectories"],
                "trajectories":   v["trajectories"],
            } for k, v in split_stats.items()},
        },

        "normalisation": normalisation,

        "files_per_split": [
            CSV_T,
            CSV_RAW_Q, CSV_RAW_QD, CSV_RAW_QDD,
            CSV_RAW_TAU_MEASURED, CSV_RAW_TAU_DECOMPOSED,
            CSV_FILTERED_Q, CSV_FILTERED_QD, CSV_FILTERED_QDD,
            CSV_FILTERED_TAU_MEASURED, CSV_FILTERED_TAU_DECOMPOSED,
            "trajectories_catalog.csv",
        ],
    }

    meta_path = os.path.join(run_dir, METADATA_FILE)
    with open(meta_path, "w") as f:
        _json.dump(meta_doc, f, indent=2, default=str)

    _log(f"Metadata saved -> {meta_path}")

    return {"meta_path": meta_path, "log_lines": log_lines}
