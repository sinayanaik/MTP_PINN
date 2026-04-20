"""Checkpoint files, run IDs, YAML metadata, and exhaustive hyperparameter merge."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

CHECKPOINT_SCHEMA_VERSION = 1
BEST_CKPT_NAME = "model.pt"
FINAL_CKPT_NAME = "model_final.pt"


def exhaustive_hparams(hp: dict[str, Any], default_base: dict[str, Any]) -> dict[str, Any]:
    full = dict(default_base)
    for k, v in (hp or {}).items():
        if str(k).startswith("_"):
            continue
        full[k] = v
    return full


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {_make_serializable(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    if hasattr(obj, "__class__") and "torch" in type(obj).__module__:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def dump_yaml(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(obj, f, Dumper=NoAliasDumper, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _fmt_hp_value(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "-".join(
            str(int(x)) if isinstance(x, (int, float)) and float(x).is_integer() else str(x) for x in v
        )
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")
        if float(v).is_integer():
            return str(int(v))
        return f"{v:g}"
    return str(v)


def build_run_id(
    model_type: str,
    *,
    epochs_trained: int,
    rmse: float,
    hp: dict[str, Any] | None,
    run_id_hp_keys: list[tuple[str, str]],
    timestamp: str | None = None,
) -> str:
    parts = [model_type, f"ep{int(epochs_trained)}", f"rmse{float(rmse):.5f}"]
    hp = hp or {}
    for key, prefix in run_id_hp_keys:
        if key not in hp or hp[key] is None:
            continue
        parts.append(f"{prefix}{_fmt_hp_value(hp[key])}")
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M")
    parts.append(stamp)
    return "_".join(parts)


def save_checkpoints(
    save_dir: str,
    *,
    model: Any,
    final_state: dict,
    best_epoch: int,
    epochs_trained: int,
    model_cls_name: str,
    hparams_blob: Any,
    norm_stats: dict,
    avg_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
) -> tuple[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, BEST_CKPT_NAME)
    final_path = os.path.join(save_dir, FINAL_CKPT_NAME)
    common = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_class": model_cls_name,
        "hparams": hparams_blob,
        "norm_stats": norm_stats,
        "epochs_trained": int(epochs_trained),
    }
    torch.save(
        {
            **common,
            "model_state": model.state_dict(),
            "checkpoint_kind": "best",
            "best_epoch": int(best_epoch),
            "metrics": avg_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        best_path,
    )
    torch.save(
        {
            **common,
            "model_state": final_state,
            "checkpoint_kind": "final",
        },
        final_path,
    )
    return best_path, final_path
