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


class TestTrajectoryMacroCanonical(unittest.TestCase):
    """Lock the canonical trajectory-macro estimator.

    This is the metric used for the live per-epoch ``val_rmse``, early
    stopping, and the returned/ranked headline test RMSE.  It must NOT equal
    the pooled-per-joint ``rmse_macro_mean`` when trajectories are
    heterogeneous.  The two diverge because (a) per-trajectory sqrt-then-mean
    vs pooled sqrt and (b) equal-per-trajectory weighting vs equal-per-sample
    weighting; the *sign* of the gap is data-dependent (a short, hard
    trajectory is upweighted by macro, a long easy one downweighted).  Mixing
    the two — live val_rmse using macro, reported test using pooled — was the
    original spurious val/test gap.
    """

    @staticmethod
    def _ref_traj_macro(pred, target, trajs):
        """Independent reference: mean over trajectories of
        (mean over joints of per-joint within-trajectory RMSE)."""
        vals = []
        for t in trajs:
            s, e = t["start_idx"], t["end_idx_exclusive"]
            d = pred[s:e] - target[s:e]
            vals.append(float(np.sqrt((d**2).mean(axis=0)).mean()))
        return float(np.mean(vals))

    def test_traj_macro_differs_from_pooled_when_heterogeneous(self):
        from Neural_Networks.models.shared.metrics_numpy import (
            compute_metrics as real_compute_metrics,
            macro_rmse_numpy,
        )

        rng = np.random.default_rng(7)
        # 3 trajectories, unequal lengths, very different error magnitudes.
        t0 = rng.normal(0.0, 0.01, size=(40, 2))   # easy, long
        t1 = rng.normal(0.0, 0.50, size=(8, 2))    # hard, short
        t2 = rng.normal(0.0, 0.05, size=(20, 2))   # medium
        err = np.concatenate([t0, t1, t2], axis=0)
        target = rng.normal(0.0, 1.0, size=err.shape)
        pred = target + err
        trajs = [
            {"start_idx": 0,  "end_idx_exclusive": 40},
            {"start_idx": 40, "end_idx_exclusive": 48},
            {"start_idx": 48, "end_idx_exclusive": 68},
        ]
        got = macro_rmse_numpy(pred, target, trajs)
        ref = self._ref_traj_macro(pred, target, trajs)
        self.assertAlmostEqual(got, ref, places=9)

        pooled_per_joint = real_compute_metrics(pred, target)["rmse_macro_mean"]
        # The two estimators must genuinely diverge here (sign is
        # data-dependent — the point is they are NOT interchangeable).
        self.assertGreater(abs(got - pooled_per_joint), 1e-3)

    def test_traj_macro_converges_on_single_trajectory(self):
        from Neural_Networks.models.shared.metrics_numpy import (
            compute_metrics as real_compute_metrics,
            macro_rmse_numpy,
        )

        rng = np.random.default_rng(1)
        target = rng.normal(size=(50, 3))
        pred = target + rng.normal(0.0, 0.1, size=(50, 3))
        one = [{"start_idx": 0, "end_idx_exclusive": 50}]
        # Single trajectory ⇒ no inter-trajectory averaging ⇒ identical to
        # the pooled-per-joint mean.
        self.assertAlmostEqual(
            macro_rmse_numpy(pred, target, one),
            real_compute_metrics(pred, target)["rmse_macro_mean"],
            places=9,
        )
        # Empty trajectory list falls back to the whole array (same value).
        self.assertAlmostEqual(
            macro_rmse_numpy(pred, target, []),
            macro_rmse_numpy(pred, target, one),
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
