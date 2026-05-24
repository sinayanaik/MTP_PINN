#!/usr/bin/env python3
"""Generate the exhaustive CSV tables — one behind every figure, plus the
champion / grid scorecards. All tables store **raw, unsmoothed** data
(Savitzky–Golay is a plot-only display choice).

  champion_selection.csv        winners under global/train/val/test (—, simple)
  headline_metrics.csv          champion scorecard            (—)
  grid_summary.csv              per-arch grid statistics      (fig01)
  grid_runs.csv                 every grid run                (fig01)
  data_efficiency.csv           RMSE/val/gap vs data fraction (fig02, fig03)
  capability_profile.csv        unexplored-metric radar       (fig04)
  training_curves.csv           per-epoch champion curves     (fig05, fig06)
  edr_correction_evolution.csv  EDR δ-term magnitudes/epoch   (fig07)
  per_joint_metrics.csv         per-joint RMSE/R²/MAE/NRMSE   (fig08, fig09)
  trajectory_tracking.csv       selected-trajectory samples   (fig10)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from shared import dataio, palette
from shared.config import default_config
from shared.plotting import champion_results, per_traj_rmse, select_trajectory
from shared.tableio import save_table

CONFIG = default_config()

# Champion-selection bases reported side by side (the user wants global, train,
# val and test all compared). Each is (label, basis, full_data_only).
_SELECTION_BASES = (
    ("global", "global", False),
    ("full_data_train", "train", True),
    ("full_data_val", "val", True),
    ("full_data_test", "test", True),
)


def _champion_selection(cfg) -> pd.DataFrame:
    """Per-arch winner under each selection basis: global / train / val / test.

    A scalar scorecard (simple data) -> CSV only, no companion chart. Shows how
    the "best" model shifts with the criterion and with the data budget.
    """
    rows = []
    for sel_label, basis, fdo in _SELECTION_BASES:
        champs = dataio.champion_by_basis(basis, full_data_only=fdo)
        for a in palette.ordered_archs(cfg):
            r = champs[a]
            train_rmse, best_ep = dataio.train_at_best(str(r["run_dir"]))
            rows.append({
                "selection_basis": sel_label,
                "architecture": palette.label(cfg, a),
                "data_fraction": r["data_train_fraction"],
                "params": dataio.param_count(r["run_dir"]),
                "train_rmse": train_rmse,
                "val_rmse": r["val_rmse"],
                "test_rmse": r["test_rmse"],
                "best_val_epoch": best_ep,
                "run_id": r["run_id"],
            })
    return pd.DataFrame(rows)


def _headline(cfg, res) -> pd.DataFrame:
    champs = dataio.champions(cfg)
    rows = []
    for a in palette.ordered_archs(cfg):
        ch = champs[a]
        m = res[a]["metrics"]
        ptr = per_traj_rmse(res[a]["pred"], res[a]["target"], res[a]["traj"])
        row = {
            "architecture": palette.label(cfg, a),
            "test_rmse_traj_macro": ch["test_rmse"],
            "test_rmse_pooled": m["rmse_pooled"],
            "r2_overall": m["r2_overall"],
            "r2_mean": m["r2_mean"],
        }
        for j, name in enumerate(cfg.joint_names):
            row[f"rmse_{name}"] = m["rmse"][j]
        row["params"] = dataio.param_count(ch["run_dir"])
        # Training wall-time was not recorded for these runs; epochs_ran is the
        # available training-cost proxy.
        row["epochs_ran"] = ch["epochs_ran"]
        row["worst_traj_rmse"] = float(np.max(ptr))
        rows.append(row)
    return pd.DataFrame(rows)


def _grid_summary(cfg, g) -> pd.DataFrame:
    rows = []
    for a in palette.ordered_archs(cfg):
        v = g.loc[g["arch"] == a, "test_rmse"].to_numpy(float)
        rows.append({
            "architecture": palette.label(cfg, a),
            "n_runs": int(v.size),
            "best_test_rmse": float(v.min()),
            "mean_test_rmse": float(v.mean()),
            "std_test_rmse": float(v.std(ddof=1)),
            "median_test_rmse": float(np.median(v)),
        })
    return pd.DataFrame(rows)


def _grid_runs(cfg, g) -> pd.DataFrame:
    cols = ["n", "arch", "status", "test_rmse", "elapsed_sec",
            "data_train_fraction", "seed"]
    d = g[cols].copy()
    d["arch"] = d["arch"].map(lambda a: palette.label(cfg, a))
    return d.sort_values(["arch", "test_rmse"]).reset_index(drop=True)


def _data_efficiency(cfg) -> pd.DataFrame:
    sw = dataio.sweep_df()
    rows = []
    for a in palette.ordered_archs(cfg):
        d = sw[sw["arch"] == a].sort_values("data_train_fraction")
        for _, r in d.iterrows():
            rows.append({
                "architecture": palette.label(cfg, a),
                "data_fraction_pct": r["data_train_fraction"] * 100.0,
                "test_rmse": r["test_rmse"],          # seed-mean
                "test_rmse_std": r.get("test_rmse_std", 0.0),
                "n_seeds": int(r.get("n_seeds", 1)),
                "val_rmse": r["val_rmse"],
                "gen_gap": r["test_rmse"] - r["val_rmse"],
            })
    return pd.DataFrame(rows)


def _capability_profile(cfg, res) -> pd.DataFrame:
    """Raw + normalised values behind the fig04 capability radar.

    Five per-arch scalars that appear on no other figure (pooled RMSE, R²
    overall, mean MAE, mean NRMSE, parameter count). Normalisation matches the
    radar exactly — imported from it so the table can never drift from the plot
    (ratio-to-best for the accuracy/fit axes, log score for parameters).
    """
    import fig04_capability_radar as radar

    champs = dataio.champions(cfg)
    archs = palette.ordered_archs(cfg)
    params = {a: float(dataio.param_count(champs[a]["run_dir"])) for a in archs}

    def mean(x):
        return float(np.mean(np.asarray(x, dtype=float)))

    specs = [
        ("pooled_rmse", True, "ratio",
         [res[a]["metrics"]["rmse_pooled"] for a in archs]),
        ("r2_overall", False, "ratio",
         [res[a]["metrics"]["r2_overall"] for a in archs]),
        ("mean_mae", True, "ratio",
         [mean(res[a]["metrics"]["mae"]) for a in archs]),
        ("mean_nrmse", True, "ratio",
         [mean(res[a]["metrics"]["nrmse"]) for a in archs]),
        ("params", True, "param_log", [params[a] for a in archs]),
    ]
    rows = []
    for axis, lower, scale, raw in specs:
        raw = np.asarray(raw, dtype=float)
        if scale == "param_log":
            score = radar._param_log_score(raw, radar.PARAM_FLOOR)
        else:
            score = radar._ratio_to_best(raw, lower)
        for a, rv, sv in zip(archs, raw, score):
            rows.append({
                "architecture": palette.label(cfg, a),
                "axis": axis,
                "raw_value": float(rv),
                "norm_score": float(sv),
                "lower_is_better": bool(lower),
            })
    return pd.DataFrame(rows)


def _training_curves(cfg) -> pd.DataFrame:
    champs = dataio.champions(cfg)
    frames = []
    for a in palette.ordered_archs(cfg):
        h = dataio.load_history(champs[a]["run_dir"])
        frames.append(pd.DataFrame({
            "architecture": palette.label(cfg, a),
            "epoch": h["epoch"],
            "train_rmse": h["train_rmse"],
            "val_rmse": h["val_rmse"],
            "overfit_gap": h["val_rmse"] - h["train_rmse"],
        }))
    return pd.concat(frames, ignore_index=True)


def _edr_corrections(cfg) -> pd.DataFrame:
    edr = dataio.champions(cfg)["edr"]
    h = dataio.load_history(edr["run_dir"])
    return h[["epoch", "mean_abs_delta_g", "mean_frob_delta_M",
              "mean_abs_delta_C_qd", "mean_abs_delta_tau_f"]].copy()


def _per_joint(cfg, res) -> pd.DataFrame:
    rows = []
    for a in palette.ordered_archs(cfg):
        m = res[a]["metrics"]
        for j, name in enumerate(cfg.joint_names):
            rows.append({
                "architecture": palette.label(cfg, a),
                "joint": name,
                "rmse": m["rmse"][j],
                "r2": m["r2"][j],
                "mae": m["mae"][j],
                "nrmse": m["nrmse"][j],
            })
    return pd.DataFrame(rows)


def _trajectory(cfg, res) -> pd.DataFrame:
    s, e, geom = select_trajectory(res["edr"]["traj"], cfg)
    n = e - s
    rows = []
    for j, name in enumerate(cfg.joint_names):
        for k in range(n):
            rows.append({
                "geometry": geom,
                "sample": k,
                "joint": name,
                "measured": res["edr"]["target"][s + k, j],
                "pred_fnn": res["fnn"]["pred"][s + k, j],
                "pred_physreg": res["physreg"]["pred"][s + k, j],
                "pred_edr": res["edr"]["pred"][s + k, j],
            })
    return pd.DataFrame(rows)


def main(cfg=CONFIG) -> list[Path]:
    res = champion_results(cfg)
    g = dataio.grid_df()
    g = g[g["status"] == "ok"]
    out = [
        save_table(_champion_selection(cfg), "champion_selection", cfg),
        save_table(_headline(cfg, res), "headline_metrics", cfg),
        save_table(_grid_summary(cfg, g), "grid_summary", cfg),
        save_table(_grid_runs(cfg, g), "grid_runs", cfg),
        save_table(_data_efficiency(cfg), "data_efficiency", cfg),
        save_table(_capability_profile(cfg, res), "capability_profile", cfg),
        save_table(_training_curves(cfg), "training_curves", cfg),
        save_table(_edr_corrections(cfg), "edr_correction_evolution", cfg),
        save_table(_per_joint(cfg, res), "per_joint_metrics", cfg),
        save_table(_trajectory(cfg, res), "trajectory_tracking", cfg),
    ]
    return out


if __name__ == "__main__":
    for p in main():
        print(p)
