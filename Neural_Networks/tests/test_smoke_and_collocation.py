"""Smoke tests for the two-model registry (CPU)."""

from __future__ import annotations

import unittest

import torch

from Neural_Networks.apps.hp_registry import get_default_hp
from Neural_Networks.apps.builder import build_model
from Neural_Networks.models import MODEL_REGISTRY, PHYSICS_INPUT_MODELS


class TestSmokeModels(unittest.TestCase):
    def test_forward_each_registered_model(self):
        device = torch.device("cpu")
        for name in MODEL_REGISTRY:
            with self.subTest(model=name):
                hp = get_default_hp(name)
                m = build_model(name, hp, device)
                m.eval()
                feat = torch.randn(4, 15)
                phy = torch.randn(4, 20)

                with torch.no_grad():
                    if name in PHYSICS_INPUT_MODELS:
                        out = m(feat, phy)
                    else:
                        out = m(feat)
                self.assertEqual(out.shape[-1], 5)


class TestPhysicsRegularizedCalibBackward(unittest.TestCase):
    def test_tau_calib_backward(self):
        """Calibration + data loss backward on tiny batch."""
        from Neural_Networks.models import PhysicsRegularizedFNN

        m = PhysicsRegularizedFNN(n_joints=5, hidden_layers=[32, 32], dropout=0.0)
        m.train()
        x = torch.randn(3, 15, requires_grad=False)
        tau_hat = m(x)
        tau_p = torch.randn(3, 20)
        losses = m.compute_loss(tau_hat, torch.randn(3, 5), tau_p)
        loss = losses["data"] + losses["physics"]
        loss.backward()
        self.assertIsNotNone(m.net[0].weight.grad)
        self.assertIsNotNone(m.tau_calib.raw_scale.grad)


if __name__ == "__main__":
    unittest.main()
