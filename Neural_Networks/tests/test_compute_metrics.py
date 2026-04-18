"""Tests for pooled vs macro RMSE and robust Pearson in compute_metrics."""
from __future__ import annotations

import numpy as np

from Neural_Networks.core.metrics import compute_metrics, pooled_rmse_numpy


def test_pooled_rmse_matches_flat_mse_sqrt():
    pred = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    tgt = np.zeros((2, 2), dtype=np.float64)
    expected = float(np.sqrt(np.mean((pred - tgt) ** 2)))
    assert abs(pooled_rmse_numpy(pred, tgt) - expected) < 1e-9


def test_macro_rmse_differs_from_pooled_when_joint_scales_differ():
    # Joint 0: errors 1,1 → RMSE_j0 = 1. Joint 1: errors 3,3 → RMSE_j1 = 3.
    # Macro mean RMSE = 2. Pooled RMSE = sqrt(mean([1,1,9,9])) = sqrt(5).
    pred = np.array([[1.0, 3.0], [1.0, 3.0]], dtype=np.float64)
    tgt = np.zeros((2, 2), dtype=np.float64)
    m = compute_metrics(pred, tgt)
    assert abs(m["rmse_mean"] - 2.0) < 1e-6
    assert abs(m["rmse_pooled"] - float(np.sqrt(5.0))) < 1e-6
    assert m["rmse_macro_mean"] == m["rmse_mean"]


def test_pearson_finite_on_constant_target():
    pred = np.random.default_rng(0).standard_normal((100, 5)).astype(np.float64)
    target = np.zeros((100, 5), dtype=np.float64)
    m = compute_metrics(pred, target)
    assert all(np.isfinite(m["pearson_r"]))
    assert np.isfinite(m["pearson_r_mean"])


def test_r2_overall_defined():
    pred = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    tgt = np.array([[0.1, 0.9], [2.1, 2.9]], dtype=np.float64)
    m = compute_metrics(pred, tgt)
    assert "r2_overall" in m and np.isfinite(m["r2_overall"])
