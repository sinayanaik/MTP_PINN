"""Enrich records with training-history info and (optionally) train-split metrics.

`mode` is the contract from analyzer.prompt.recompute_mode:
  - "all"    : recompute every model's train metrics (force=True)
  - "missing": compute only models without a current cache
  - "cached" : read caches if present; never touch torch
  - "skip"   : leave train_metrics empty
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from ..io.history import load_history, best_epoch_info

logger = logging.getLogger(__name__)

Mode = Literal["all", "missing", "cached", "skip"]


def enrich_records(records: list[dict[str, Any]], mode: Mode = "missing") -> None:
    """Mutate records in place with _history, _best_epoch, train_metrics."""
    for rec in records:
        hist = load_history(rec.get("_history_path"))
        rec["_history"] = hist
        ep, tr, vr = best_epoch_info(hist)
        rec["_best_epoch"] = ep
        rec["_train_rmse_hist"] = tr
        rec["_val_rmse_hist"] = vr

    if mode == "skip":
        for rec in records:
            rec["train_metrics"] = {}
        return

    if mode == "cached":
        from .train_metrics import read_cache
        for rec in records:
            rec["train_metrics"] = read_cache(rec)
        return

    # "all" or "missing" — both may invoke the (lazy-torch) compute path
    from .train_metrics import compute, has_cache
    for rec in records:
        if mode == "missing" and has_cache(rec):
            from .train_metrics import read_cache
            rec["train_metrics"] = read_cache(rec)
        else:
            rec["train_metrics"] = compute(rec, force=(mode == "all"))

    _log_metric_scale_check(records)


def _log_metric_scale_check(records: list[dict[str, Any]]) -> None:
    """First record with both train+test populated: log side-by-side comparison."""
    for rec in records:
        tm = rec.get("train_metrics") or {}
        te = rec.get("test_metrics") or {}
        if tm.get("rmse_pooled") is not None and te.get("rmse_pooled") is not None:
            logger.info(
                "Metric scale check (%s): train rmse_pooled=%.5f / test=%.5f  "
                "train nrmse_mean=%.5f / test=%.5f",
                Path(rec.get("_run_dir", "?")).name,
                float(tm.get("rmse_pooled", float("nan"))),
                float(te.get("rmse_pooled", float("nan"))),
                float(tm.get("nrmse_mean", float("nan"))),
                float(te.get("nrmse_mean", float("nan"))),
            )
            break
