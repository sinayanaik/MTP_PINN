"""Smoke tests for packaged torque models (CPU)."""

from __future__ import annotations

import unittest

import torch

from Neural_Networks.models import BlackBoxFNN, reduce_physics_to_total


class TestSmokeModels(unittest.TestCase):
    def test_blackbox_forward_from_models(self):
        m = BlackBoxFNN(hidden_layers=[16, 16], dropout=0.0)
        m.eval()
        feat = torch.randn(4, 15)
        phy = torch.randn(4, 20)
        with torch.no_grad():
            out = m(feat, phy)
        self.assertEqual(out.shape, (4, 5))

    def test_physics_sum_shape(self):
        phy = torch.randn(4, 20)
        s = reduce_physics_to_total(phy)
        self.assertEqual(s.shape, (4, 5))


if __name__ == "__main__":
    unittest.main()
