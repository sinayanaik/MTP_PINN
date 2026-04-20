"""NumPy regression metrics for torque evaluation (physical N·m space)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def macro_rmse_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    trajectories: list[dict],
) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = len(pred)
    if not trajectories:
        slices = [(0, n)]
    else:
        slices = []
        for traj in trajectories:
            s, e = int(traj["start_idx"]), int(traj["end_idx_exclusive"])
            if e <= s or s >= n:
                continue
            slices.append((s, min(e, n)))
        if not slices:
            slices = [(0, n)]
    per_traj: list[float] = []
    for s, e in slices:
        diff = pred[s:e] - target[s:e]
        rmse_j = np.sqrt((diff**2).mean(axis=0))
        per_traj.append(float(rmse_j.mean()))
    return float(np.mean(per_traj))


def _pearson_r_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    diff = pred - target
    mse = (diff**2).mean(axis=0)
    rmse = np.sqrt(mse)
    mae = np.abs(diff).mean(axis=0)
    max_e = np.abs(diff).max(axis=0)
    t_range = target.max(axis=0) - target.min(axis=0)
    nrmse = rmse / (t_range + 1e-8)
    ss_res = (diff**2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)
    pearson_r = np.array([_pearson_r_safe(pred[:, j], target[:, j]) for j in range(pred.shape[1])])
    exp_var = 1.0 - np.var(diff, axis=0) / (np.var(target, axis=0) + 1e-10)
    mse_pooled = float(np.mean(diff.astype(np.float64, copy=False) ** 2))
    rmse_pooled = float(math.sqrt(mse_pooled))
    pv = pred.reshape(-1).astype(np.float64, copy=False)
    tv = target.reshape(-1).astype(np.float64, copy=False)
    ss_res_all = float(np.sum((pv - tv) ** 2))
    ss_tot_all = float(np.sum((tv - tv.mean()) ** 2))
    r2_overall = float(1.0 - ss_res_all / (ss_tot_all + 1e-10))
    return {
        "mse": mse.tolist(),
        "rmse": rmse.tolist(),
        "nrmse": nrmse.tolist(),
        "mae": mae.tolist(),
        "max_error": max_e.tolist(),
        "r2": r2.tolist(),
        "pearson_r": pearson_r.tolist(),
        "explained_variance": exp_var.tolist(),
        "mse_mean": float(mse.mean()),
        "rmse_mean": float(rmse.mean()),
        "rmse_macro_mean": float(rmse.mean()),
        "mse_pooled": mse_pooled,
        "rmse_pooled": rmse_pooled,
        "r2_overall": r2_overall,
        "nrmse_mean": float(nrmse.mean()),
        "mae_mean": float(mae.mean()),
        "r2_mean": float(r2.mean()),
        "pearson_r_mean": float(np.mean(pearson_r)),
    }
