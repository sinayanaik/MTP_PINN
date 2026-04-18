"""Tests for pooled vs macro RMSE and robust Pearson in compute_metrics."""
from __future__ import annotations

import unittest

import numpy as np

from Neural_Networks.apps.metrics import compute_metrics, pooled_rmse_numpy


class TestComputeMetrics(unittest.TestCase):
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
