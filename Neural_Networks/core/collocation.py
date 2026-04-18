"""
Physics collocation for Equation-Constrained PINNs: sample (q, q̇, q̈), RNEA+friction, match τ̂.

Uses the same normalisation as ``RobotDataset`` (train-split metadata).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from Neural_Networks.core.physics import (
    ACTIVE_JOINTS,
    build_pinocchio_model,
    compute_rnea_decomposition,
    torque_friction,
)

if TYPE_CHECKING:
    from Neural_Networks.data.loader import RobotDataset

logger = logging.getLogger(__name__)


class CollocationSampler:
    """
    Sample synthetic states in physical units, run Pinocchio RNEA + friction,
    return normalised features and normalised analytical torque (same as CSV targets).
    """

    def __init__(self, train_dataset: "RobotDataset"):
        self._nj = ACTIVE_JOINTS
        norm = train_dataset.metadata.get("normalisation", {})
        self._mq = np.asarray(norm["mean_q"], dtype=np.float64)
        self._sq = np.asarray(norm["std_q"], dtype=np.float64).clip(min=1e-8)
        self._mqd = np.asarray(norm["mean_qd"], dtype=np.float64)
        self._sqd = np.asarray(norm["std_qd"], dtype=np.float64).clip(min=1e-8)
        self._mqdd = np.asarray(norm["mean_qdd"], dtype=np.float64)
        self._sqdd = np.asarray(norm["std_qdd"], dtype=np.float64).clip(min=1e-8)
        self._mt = np.asarray(norm["mean_tau"], dtype=np.float64)
        self._st = np.asarray(norm["std_tau"], dtype=np.float64).clip(min=1e-8)

        self._pin_model = None
        self._pin_data = None
        self._pin_ok = False
        try:
            self._pin_model, self._pin_data, _ = build_pinocchio_model()
            self._pin_ok = True
        except Exception as e:
            logger.warning("Collocation disabled: Pinocchio build failed (%s)", e)

    @property
    def ok(self) -> bool:
        return self._pin_ok

    def sample(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features_norm, tau_analytical_norm): features (n, 15), tau (n, n_joints)."""
        if not self._pin_ok or self._pin_model is None:
            raise RuntimeError("CollocationSampler: Pinocchio not available")

        n = max(1, int(n))
        # q: uniform in per-joint [mean - 3σ, mean + 3σ]; q̇, q̈: Gaussian from train marginals
        q_lo = self._mq - 3.0 * self._sq
        q_hi = self._mq + 3.0 * self._sq
        q = np.random.uniform(q_lo, q_hi, size=(n, self._nj)).astype(np.float32)
        qd = np.random.normal(self._mqd, self._sqd, size=(n, self._nj)).astype(np.float32)
        qdd = np.random.normal(self._mqdd, self._sqdd, size=(n, self._nj)).astype(np.float32)

        r_rnea, _, _, _ = compute_rnea_decomposition(
            self._pin_model, self._pin_data, q, qd, qdd, n_active=self._nj
        )
        tau_f = torque_friction(qd)
        tau_phys = r_rnea + tau_f

        qn = (q - self._mq.astype(np.float32)) / self._sq.astype(np.float32)
        qdn = (qd - self._mqd.astype(np.float32)) / self._sqd.astype(np.float32)
        qddn = (qdd - self._mqdd.astype(np.float32)) / self._sqdd.astype(np.float32)
        feat = np.concatenate([qn, qdn, qddn], axis=-1)

        taun = (tau_phys.astype(np.float64) - self._mt) / self._st

        f = torch.from_numpy(feat).to(device=device, dtype=torch.float32)
        t = torch.from_numpy(taun.astype(np.float32)).to(device=device, dtype=torch.float32)
        return f, t
