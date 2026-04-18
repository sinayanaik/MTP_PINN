"""Smoke tests for model registry and EC collocation-style backward (CPU)."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from Neural_Networks.models import (
    MODEL_REGISTRY,
    DECOMPOSED_MODELS,
    LAGRANGIAN_MODELS,
    PHYSICS_INPUT_MODELS,
)
from Neural_Networks.models.equation_constrained_pinn_fnn import EquationConstrainedPINNFNN
from Neural_Networks.core import build_model
from Neural_Networks.apps.train import get_default_hp


class TestSmokeModels(unittest.TestCase):
    def test_forward_each_registered_model(self):
        device = torch.device("cpu")
        for name in MODEL_REGISTRY:
            with self.subTest(model=name):
                hp = get_default_hp(name)
                m = build_model(name, hp, device)
                m.eval()
                feat = torch.randn(4, 15)
                phy = torch.randn(4, 5)

                with torch.no_grad():
                    if name in PHYSICS_INPUT_MODELS:
                        out = m(feat, phy)
                    elif name in DECOMPOSED_MODELS:
                        out, _c = m(feat, phy)
                    elif name in LAGRANGIAN_MODELS:
                        out, _c = m(feat)
                    else:
                        out = m(feat)
                self.assertEqual(out.shape[-1], 5)


class TestEquationCollocationBackward(unittest.TestCase):
    def test_tau_calib_and_prediction_backward(self):
        """Residual + calib MSE backward on tiny batch (no Pinocchio)."""
        m = EquationConstrainedPINNFNN(n_joints=5, hidden_layers=[32, 32], dropout=0.0)
        m.train()
        x = torch.randn(3, 15, requires_grad=False)
        tau_p_n = torch.randn(3, 5)
        tau_hat = m(x)
        tau_eff = m.tau_calib(tau_p_n)
        loss = F.mse_loss(tau_hat, tau_eff)
        loss.backward()
        self.assertIsNotNone(m.net[0].weight.grad)
        self.assertIsNotNone(m.tau_calib.raw_scale.grad)


if __name__ == "__main__":
    unittest.main()
