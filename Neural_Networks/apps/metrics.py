"""
Neural_Networks.core.metrics
==============================
Regression metrics for torque prediction evaluation.

All functions operate on raw NumPy arrays in **physical units (N·m)**.
They have no UI or I/O side effects — outputs are plain Python dicts or floats.

These functions are the single canonical source of truth for evaluation
metrics; both the training pipeline (core/trainer.py) and the GUI
visualizer (gui/visualizer_app.py) import from here.

Public API
----------
pooled_rmse_numpy(pred, target)        -> float
trajectory_mean_rmse_numpy(pred, target, trajectories) -> float
compute_metrics(pred, target)          -> dict
"""

from __future__ import annotations

import math

import numpy as np


def pooled_rmse_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """Unweighted RMSE over all elements — single pooled scalar.

    Equivalent to ``sqrt(mean((pred − target)²))`` flattened over every
    sample and joint simultaneously.  This matches the ``val_rmse`` metric
    used for early stopping and checkpoint selection.
    """
    d = pred.astype(np.float64, copy=False) - target.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(d * d)))


def trajectory_mean_rmse_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    trajectories: list[dict],
) -> float:
    """Macro-average RMSE: mean of per-trajectory RMSEs.

    Each trajectory segment [start_idx, end_idx_exclusive) is evaluated
    independently and the results averaged, so a short high-error trajectory
    contributes equally to a long easy one.  This is the canonical validation
    metric; it prevents a single long trajectory from dominating.

    Falls back to pooled_rmse_numpy when no trajectory boundaries are
    available (empty list or out-of-range indices).

    Parameters
    ----------
    pred, target : np.ndarray  shape (N, J)
    trajectories : list of dicts with keys ``start_idx``, ``end_idx_exclusive``
    """
    if not trajectories:
        return pooled_rmse_numpy(pred, target)
    n = len(pred)
    per_traj: list[float] = []
    for traj in trajectories:
        s, e = traj["start_idx"], traj["end_idx_exclusive"]
        if e <= s or s >= n:
            continue
        e = min(e, n)
        per_traj.append(pooled_rmse_numpy(pred[s:e], target[s:e]))
    return float(np.mean(per_traj)) if per_traj else pooled_rmse_numpy(pred, target)


def macro_rmse_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    trajectories: list[dict],
) -> float:
    """Canonical val_rmse: mean over trajectories of mean-per-joint RMSE.

    Algorithm
    ---------
    For each trajectory slice [s, e):
        diff   = pred[s:e] - target[s:e]          # (T, J)
        rmse_j = sqrt(mean_t(diff^2))              # (J,) — one RMSE per joint
        traj_rmse = mean_j(rmse_j)                 # scalar — equal weight per joint
    val_rmse = mean(traj_rmse values)              # macro-average across trajectories

    Why mean-joint rather than pooled per trajectory
    ------------------------------------------------
    Robot joints have very different torque magnitudes (J2 shoulder >> J4 wrist).
    Pooled RMSE (sqrt of mean over J*T elements) lets high-torque joints dominate.
    Per-joint then averaged gives equal weight to every joint's prediction quality,
    which is the correct criterion for a robot controller.

    Fallback
    --------
    If ``trajectories`` is empty, the entire array is treated as one trajectory.

    Parameters
    ----------
    pred, target : np.ndarray  shape (N, J)  — same normalisation space
    trajectories : list of dicts with keys ``start_idx``, ``end_idx_exclusive``
    """
    pred   = np.asarray(pred,   dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n      = len(pred)

    # Build a list of (start, end) slices to evaluate
    if not trajectories:
        slices = [(0, n)]
    else:
        slices = []
        for traj in trajectories:
            s, e = int(traj["start_idx"]), int(traj["end_idx_exclusive"])
            if e <= s or s >= n:
                continue
            slices.append((s, min(e, n)))
        if not slices:           # all trajectories out of range — fall back
            slices = [(0, n)]

    per_traj: list[float] = []
    for s, e in slices:
        diff     = pred[s:e] - target[s:e]          # (T, J)
        rmse_j   = np.sqrt((diff ** 2).mean(axis=0)) # (J,) per-joint RMSE
        per_traj.append(float(rmse_j.mean()))         # mean across joints

    return float(np.mean(per_traj))


def _pearson_r_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with graceful handling of constant arrays."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute per-joint and aggregate regression metrics.

    Inputs should be in **physical units (N·m)** — de-normalise before
    calling if your arrays are in normalised space.

    Per-joint metrics (axis 0)
    --------------------------
    mse, rmse, nrmse (target range), mae, max_error, r2, pearson_r,
    explained_variance

    Aggregate metrics
    -----------------
    rmse_mean    — macro mean of per-joint RMSEs  (mean√MSEⱼ)
    rmse_pooled  — √(mean squared error over all N×J elements)
    r2_mean      — mean of per-joint R²
    r2_overall   — single R² on flattened predictions
    nrmse_mean, mae_mean, pearson_r_mean — means of per-joint values
    mse_mean, mse_pooled — corresponding MSE variants
    """
    diff  = pred - target
    mse   = (diff ** 2).mean(axis=0)
    rmse  = np.sqrt(mse)
    mae   = np.abs(diff).mean(axis=0)
    max_e = np.abs(diff).max(axis=0)

    # NRMSE normalised by per-joint range (avoids scale artefacts across joints)
    t_range = target.max(axis=0) - target.min(axis=0)
    nrmse   = rmse / (t_range + 1e-8)

    # R² per joint
    ss_res  = (diff ** 2).sum(axis=0)
    ss_tot  = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    r2      = 1.0 - ss_res / (ss_tot + 1e-10)

    pearson_r = np.array([
        _pearson_r_safe(pred[:, j], target[:, j])
        for j in range(pred.shape[1])
    ])
    exp_var = 1.0 - np.var(diff, axis=0) / (np.var(target, axis=0) + 1e-10)

    # Pooled (global) metrics
    mse_pooled  = float(np.mean(diff.astype(np.float64, copy=False) ** 2))
    rmse_pooled = float(math.sqrt(mse_pooled))
    pv = pred.reshape(-1).astype(np.float64, copy=False)
    tv = target.reshape(-1).astype(np.float64, copy=False)
    ss_res_all = float(np.sum((pv - tv) ** 2))
    ss_tot_all = float(np.sum((tv - tv.mean()) ** 2))
    r2_overall = float(1.0 - ss_res_all / (ss_tot_all + 1e-10))

    return {
        "mse":               mse.tolist(),
        "rmse":              rmse.tolist(),
        "nrmse":             nrmse.tolist(),
        "mae":               mae.tolist(),
        "max_error":         max_e.tolist(),
        "r2":                r2.tolist(),
        "pearson_r":         pearson_r.tolist(),
        "explained_variance": exp_var.tolist(),
        # Aggregate
        "mse_mean":          float(mse.mean()),
        "rmse_mean":         float(rmse.mean()),
        "rmse_macro_mean":   float(rmse.mean()),   # alias kept for backward compat
        "mse_pooled":        mse_pooled,
        "rmse_pooled":       rmse_pooled,
        "r2_overall":        r2_overall,
        "nrmse_mean":        float(nrmse.mean()),
        "mae_mean":          float(mae.mean()),
        "r2_mean":           float(r2.mean()),
        "pearson_r_mean":    float(np.mean(pearson_r)),
    }
