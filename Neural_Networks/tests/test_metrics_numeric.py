"""Self-contained regression metric tests (no training-package imports)."""

from __future__ import annotations

import math
import unittest

import numpy as np


def pooled_rmse_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    d = pred.astype(np.float64, copy=False) - target.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(d * d)))


def _pearson_r_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
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


class TestMetricsNumeric(unittest.TestCase):
    def test_pooled_rmse_matches_flat_mse_sqrt(self):
        pred = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float64)
        tgt = np.zeros((2, 2), dtype=np.float64)
        expected = float(np.sqrt(np.mean((pred - tgt) ** 2)))
        self.assertLess(abs(pooled_rmse_numpy(pred, tgt) - expected), 1e-9)

    def test_macro_rmse_differs_from_pooled_when_joint_scales_differ(self):
        pred = np.array([[1.0, 3.0], [1.0, 3.0]], dtype=np.float64)
        tgt = np.zeros((2, 2), dtype=np.float64)
        m = compute_metrics(pred, tgt)
        self.assertLess(abs(m["rmse_mean"] - 2.0), 1e-6)
        self.assertLess(abs(m["rmse_pooled"] - float(np.sqrt(5.0))), 1e-6)
        self.assertEqual(m["rmse_macro_mean"], m["rmse_mean"])

    def test_pearson_finite_on_constant_target(self):
        pred = np.random.default_rng(0).standard_normal((100, 5)).astype(np.float64)
        target = np.zeros((100, 5), dtype=np.float64)
        m = compute_metrics(pred, target)
        self.assertTrue(all(np.isfinite(m["pearson_r"])))
        self.assertTrue(np.isfinite(m["pearson_r_mean"]))

    def test_r2_overall_defined(self):
        pred = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        tgt = np.array([[0.1, 0.9], [2.1, 2.9]], dtype=np.float64)
        m = compute_metrics(pred, tgt)
        self.assertIn("r2_overall", m)
        self.assertTrue(np.isfinite(m["r2_overall"]))


if __name__ == "__main__":
    unittest.main()
