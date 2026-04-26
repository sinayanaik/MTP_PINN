"""Record-dict helpers: split accessors, label formatters, ranking."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..config import N_JOINTS, _TYPE_ABBREV


def get_split(rec: dict[str, Any], split: str) -> dict[str, Any]:
    key_map = {
        "val":        "val_metrics",
        "test":       "test_metrics",
        "train":      "train_metrics",
        "checkpoint": "metrics",
    }
    return rec.get(key_map.get(split, f"{split}_metrics"), {}) or {}


def split_scalar(
    rec: dict[str, Any], split: str, *keys: str, default: float = float("nan"),
) -> float:
    d = get_split(rec, split)
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


def rmse_scalar(rec: dict[str, Any], split: str) -> float:
    """Average RMSE across all joints (rmse_mean), falling back to rmse_pooled."""
    v = split_scalar(rec, split, "rmse_mean")
    if not (v == v and np.isfinite(v)):
        v = split_scalar(rec, split, "rmse_pooled")
    return v


def train_rmse(rec: dict[str, Any]) -> float:
    """Train RMSE: tries train_metrics first, falls back to training history."""
    v = rmse_scalar(rec, "train")
    if not (v == v and np.isfinite(v)):
        v = rec.get("_train_rmse_hist", float("nan"))
    return v if isinstance(v, float) else float("nan")


def split_joints(rec: dict[str, Any], split: str, key: str) -> list[float]:
    d = get_split(rec, split)
    v = d.get(key)
    if isinstance(v, list) and len(v) == N_JOINTS:
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            pass
    return [float("nan")] * N_JOINTS


def short_label(run_id: str) -> str:
    m = re.search(r"ep(\d+)_rmse([0-9.]+)", run_id)
    prefix = re.match(r"^([A-Za-z0-9]+)", run_id)
    if m and prefix:
        raw_type = prefix.group(1)
        abbrev = _TYPE_ABBREV.get(raw_type, raw_type[:8])
        rmse_str = m.group(2)[:7]
        return f"{abbrev} ep{m.group(1)} r{rmse_str}"
    return run_id[:30]


def arch_short_label(mtype: str) -> str:
    return {
        "BlackBoxFNN":           "Black-Box",
        "PhysicsRegularizedFNN": "Physics-Reg",
        "ResidualCorrectionFNN": "Residual-Corr",
        "EDR":                   "EDR",
    }.get(mtype, mtype[:14])


def model_label(rec: dict[str, Any]) -> str:
    return rec.get("model_type", short_label(rec.get("run_id", "?")))


def sorted_records(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    all_recs = [r for recs in groups.values() for r in recs]
    all_recs.sort(key=lambda r: rmse_scalar(r, "test"))
    return all_recs


def best_per_type(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    bests: list[dict[str, Any]] = []
    for recs in groups.values():
        best = min(recs, key=lambda r: rmse_scalar(r, "test"))
        bests.append(best)
    bests.sort(key=lambda r: rmse_scalar(r, "test"))
    return bests


def best_blackbox_record(
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    bbs = groups.get("BlackBoxFNN", [])
    if not bbs:
        return None
    return min(bbs, key=lambda r: rmse_scalar(r, "test"))
