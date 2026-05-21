#!/usr/bin/env python3
"""A3 — Friction-net ablation.

Hypothesis: EDR's val<test gap is driven by the friction sub-net overfitting
to val-trajectory q̇ distributions.  Test: shrink the friction net to a tiny
[1,1] MLP and turn off use_friction_qdd so the network can only learn a
near-constant friction correction.  If the val<test gap shrinks, friction
overfit is the culprit.

From repository root::

    PYTHONPATH=. python3 Neural_Networks/diagnostics/run_a3_friction_ablation.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_DIAG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIAG_DIR))

# Reuse the A1 harness — same data fraction, same epoch budget, same logger.
import run_a1_baseline as A1  # noqa: E402
from run_a1_baseline import _attach_q_norm_for_edr, _run_one  # noqa: E402
from edr_strategy import EDR_STRATEGY  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("diag.a3")

    hp = dict(A1.EDR_HP)
    # Ablate friction net to near-zero capacity.  The δτ_f head still exists
    # (architecture invariant) but with width 1 and no q̈ input it can only
    # output a near-constant per-joint bias.
    hp["friction_hidden"] = [1, 1]
    hp["use_friction_qdd"] = False
    _attach_q_norm_for_edr(hp)

    log.info("=" * 78)
    log.info("A3 — EDR with friction net ablated (friction_hidden=[1,1], qdd=off)")
    log.info("=" * 78)
    s = _run_one("EDR_A3_friction_ablated", EDR_STRATEGY, hp)

    log.info("[%s]  best_val=ep%d val=%.5f  test@bv=%.5f  best_test=ep%d test=%.5f",
             s["name"], s["best_val_epoch"], s["best_val_rmse"],
             s["test_rmse_at_best_val"], s["best_test_epoch"], s["best_test_rmse"])


if __name__ == "__main__":
    main()
