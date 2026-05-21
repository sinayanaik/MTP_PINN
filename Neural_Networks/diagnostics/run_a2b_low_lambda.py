#!/usr/bin/env python3
"""A2b — Verify low-λ_corr drives the val<test gap.

The journal's grid-winning EDR config uses lambda_correction_reg=0.002 (very
weak). Hypothesis: with such weak regularisation the δ-nets overfit val
specifically, producing the val<test gap the user reported on HPC.  My A2
run at λ_corr=0.01 didn't show the gap — does dropping to 0.002 bring it
back?

Long run (250 epochs / 15% data) — matches A2_long for direct comparison.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_DIAG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIAG_DIR))

import run_a1_baseline as A1                                                # noqa: E402
from run_a1_baseline import _attach_q_norm_for_edr, _run_one                # noqa: E402
from edr_strategy import EDR_STRATEGY                                       # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("diag.a2b")

    hp = dict(A1.EDR_HP)
    # Match the journal-winning config's regularisation strength.
    hp["lambda_correction_reg"] = 0.002
    _attach_q_norm_for_edr(hp)

    log.info("=" * 78)
    log.info("A2b — EDR with λ_corr=0.002 (journal-winning low-reg)")
    log.info("Epochs=%d  data_train_fraction=%.2f", A1._EPOCHS, A1._FRAC)
    log.info("=" * 78)
    s = _run_one("EDR_A2b_lowreg", EDR_STRATEGY, hp)

    log.info("[%s]  best_val=ep%d val=%.5f  test@bv=%.5f  best_test=ep%d test=%.5f  "
             "gap_val_test=%.5f",
             s["name"], s["best_val_epoch"], s["best_val_rmse"],
             s["test_rmse_at_best_val"], s["best_test_epoch"], s["best_test_rmse"],
             s["test_rmse_at_best_val"] - s["best_val_rmse"])


if __name__ == "__main__":
    main()
