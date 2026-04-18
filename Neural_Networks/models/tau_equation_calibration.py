"""Learnable affine calibration of nominal equation torque τ_eq (per-joint scale + bias)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TauEquationCalibration(nn.Module):
    """
    τ_eq_eff = diag(s) τ_eq + b, with s_j = softplus(z_j) + ε (positive scales).

    Initialised so s ≈ 1 and b = 0 (softplus(log(e−1)) = 1).
    """

    def __init__(self, n_joints: int, eps: float = 1e-5):
        super().__init__()
        inv = math.log(math.e - 1.0)
        self.raw_scale = nn.Parameter(torch.full((n_joints,), inv))
        self.bias = nn.Parameter(torch.zeros(n_joints))
        self.eps = eps

    def forward(self, tau_eq: torch.Tensor) -> torch.Tensor:
        s = F.softplus(self.raw_scale) + self.eps
        return s * tau_eq + self.bias
