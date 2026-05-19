#!/usr/bin/env python3
"""Generate the exhaustive CSV tables — one behind every figure, plus the
champion / grid scorecards. All tables store **raw, unsmoothed** data
(Savitzky–Golay is a plot-only display choice).

  headline_metrics.csv          champion scorecard            (—)
  grid_summary.csv              per-arch grid statistics      (fig01)
  grid_runs.csv                 every grid run                (fig01)
  data_efficiency.csv           RMSE/val/gap vs data fraction (fig02, fig03)
  cost_accuracy.csv             params/time vs RMSE           (fig04, fig05)
  training_curves.csv           per-epoch champion curves     (fig06, fig07)
  edr_correction_evolution.csv  EDR δ-term magnitudes/epoch   (fig08)
  per_joint_metrics.csv         per-joint RMSE/R²/MAE/NRMSE   (fig09, fig10)
  trajectory_tracking.csv       selected-trajectory samples   (fig11)
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


def _headline(cfg, res) -> pd.DataFrame:
    champs = dataio.champions(cfg.champion_metric)
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
        row["train_seconds"] = ch["time_seconds"]
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


def _cost_accuracy(cfg) -> pd.DataFrame:
    champ_ids = {r["run_id"] for r in
                 dataio.champions(cfg.champion_metric).values()}
    rows = []
    for r in dataio.registry_records():
        try:
            p = dataio.param_count(r["run_dir"])
        except Exception:  # noqa: BLE001
            continue
        rows.append({
            "architecture": palette.label(cfg, r["arch"]),
            "run_id": r["run_id"],
            "params": p,
            "train_seconds": r["time_seconds"],
            "train_minutes": r["time_seconds"] / 60.0,
            "test_rmse": r["test_rmse"],
            "is_champion": r["run_id"] in champ_ids,
        })
    return pd.DataFrame(rows).sort_values(
        ["architecture", "test_rmse"]).reset_index(drop=True)


def _training_curves(cfg) -> pd.DataFrame:
    champs = dataio.champions(cfg.champion_metric)
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
    edr = dataio.champions(cfg.champion_metric)["edr"]
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
        save_table(_headline(cfg, res), "headline_metrics", cfg),
        save_table(_grid_summary(cfg, g), "grid_summary", cfg),
        save_table(_grid_runs(cfg, g), "grid_runs", cfg),
        save_table(_data_efficiency(cfg), "data_efficiency", cfg),
        save_table(_cost_accuracy(cfg), "cost_accuracy", cfg),
        save_table(_training_curves(cfg), "training_curves", cfg),
        save_table(_edr_corrections(cfg), "edr_correction_evolution", cfg),
        save_table(_per_joint(cfg, res), "per_joint_metrics", cfg),
        save_table(_trajectory(cfg, res), "trajectory_tracking", cfg),
    ]
    return out


if __name__ == "__main__":
    for p in main():
        print(p)
