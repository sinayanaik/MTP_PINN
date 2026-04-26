"""Load training_history.csv and identify the best-checkpoint epoch."""

from __future__ import annotations

import csv
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def load_history(path: str | None) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    if not path or not os.path.isfile(path):
        return result
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, val in row.items():
                    try:
                        result.setdefault(key, []).append(float(val))
                    except (ValueError, TypeError):
                        result.setdefault(key, []).append(float("nan"))
    except OSError as exc:
        logger.warning("Could not read history %s: %s", path, exc)
    return result


def best_epoch_info(history: dict[str, list[float]]) -> tuple[int, float, float]:
    val_rmse = history.get("val_rmse", [])
    if not val_rmse:
        return (-1, float("nan"), float("nan"))
    best_idx = int(np.nanargmin(val_rmse))
    epochs = history.get("epoch", [])
    epoch_num = int(epochs[best_idx]) if best_idx < len(epochs) else best_idx + 1
    tr_list = history.get("train_rmse", [])
    tr = tr_list[best_idx] if best_idx < len(tr_list) else float("nan")
    return (epoch_num, tr, float(val_rmse[best_idx]))
