"""Checkpoint files, run IDs, YAML metadata, and exhaustive hyperparameter merge."""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

CHECKPOINT_SCHEMA_VERSION = 1
BEST_CKPT_NAME = "model.pt"
FINAL_CKPT_NAME = "model_final.pt"

# Full training state for mid-run resume (segmented training).
TRAINING_STATE_SCHEMA = 1
TRAINING_STATE_NAME = "training_state.pt"


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


# --- Segmented / resumable training (full optimiser, scheduler, AMP, history) ----


def get_rng_state_bundle() -> dict[str, Any]:
    """Capture PyTorch, CUDA, NumPy, and stdlib random state for mid-run resume."""
    out: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def set_rng_state_bundle(bundle: dict[str, Any] | None) -> None:
    if not bundle:
        return
    if "torch" in bundle:
        torch.set_rng_state(bundle["torch"])
    if "torch_cuda" in bundle and bundle["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(bundle["torch_cuda"])
    if "numpy" in bundle:
        np.random.set_state(bundle["numpy"])
    if "python" in bundle:
        random.setstate(bundle["python"])


def collect_model_training_extras(model: Any) -> dict[str, Any]:
    """Optional EDR (or future) state not in state_dict (phase, val history)."""
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    out: dict[str, Any] = {}
    if hasattr(m, "_val_rmse_history"):
        out["val_rmse_history"] = [float(x) for x in m._val_rmse_history]
    if hasattr(m, "phase"):
        try:
            out["phase"] = int(m.phase)
        except (TypeError, ValueError):
            pass
    return out


def apply_model_training_extras(model: Any, extras: dict[str, Any] | None) -> None:
    if not extras:
        return
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    if "val_rmse_history" in extras and hasattr(m, "_val_rmse_history"):
        m._val_rmse_history.clear()
        m._val_rmse_history.extend(float(x) for x in extras["val_rmse_history"])
    if "phase" in extras and hasattr(m, "set_phase"):
        m.set_phase(int(extras["phase"]))


def save_training_state(
    path: str,
    *,
    schema: int = TRAINING_STATE_SCHEMA,
    next_epoch: int,
    epochs_max: int,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    onecycle_sched: Any,
    scaler: Any,
    history: dict[str, list],
    best_state: dict | None,
    best_epoch_num: int,
    patience_counter: int,
    best_val_loss: float,
    best_val_rmse: float,
    best_val_loss_track: float,
    best_val_rmse_phys: float,
    stopped_early: bool,
) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    out: dict[str, Any] = {
        "schema": int(schema),
        "next_epoch": int(next_epoch),
        "epochs_max": int(epochs_max),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "onecycle_state": None if onecycle_sched is None else onecycle_sched.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "history": {k: list(v) for k, v in history.items()},
        "best_state": best_state,
        "best_epoch_num": int(best_epoch_num),
        "patience_counter": int(patience_counter),
        "best_val_loss": float(best_val_loss),
        "best_val_rmse": float(best_val_rmse),
        "best_val_loss_track": float(best_val_loss_track),
        "best_val_rmse_phys": float(best_val_rmse_phys),
        "stopped_early": bool(stopped_early),
        "rng": get_rng_state_bundle(),
        "model_extras": collect_model_training_extras(model),
    }
    torch.save(out, path)


def load_training_state(path: str, map_location: str | torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=map_location)
