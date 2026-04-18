"""Uniform checkpoint + hparams serialisation used by every trainer.

Every training pipeline that calls ``Neural_Networks.apps.trainer.train_model``
MUST go through ``save_checkpoints`` so that:

  - The on-disk schema of ``model.pt`` and ``model_final.pt`` is identical
    across code paths — downstream loaders need only one reader.
  - The best-epoch and final-epoch checkpoints are always written together.
  - ``exhaustive_hparams()`` produces a complete hp dict (model defaults
    overlaid with the actual run values) so ``metadata.yaml`` records every
    knob that affected training — not just the ones the user typed.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import torch
import yaml


# File name constants — referenced everywhere that reads a checkpoint.
BEST_CKPT_NAME  = "model.pt"
FINAL_CKPT_NAME = "model_final.pt"

# Bump when the on-disk dict layout changes in an incompatible way.
CHECKPOINT_SCHEMA_VERSION = 1


def exhaustive_hparams(
    model_type: str,
    hp: dict,
    *,
    n_train_samples: int = 0,
) -> dict:
    """Return every known hp for ``model_type`` overlaid with ``hp``.

    Starts from ``get_default_hp`` (which already folds profile defaults,
    wizard defaults and model-specific extras) so the result reflects
    *every* knob the trainer could have read, not just the keys the caller
    happened to pass in.  Private keys starting with ``_`` are dropped to
    keep the metadata readable.
    """
    try:
        from Neural_Networks.apps.hp_registry import get_default_hp
        epochs = int(hp.get("epochs", 500) or 500)
        full = get_default_hp(model_type,
                              n_train_samples=int(n_train_samples or 0),
                              epochs=epochs)
    except Exception:
        full = {}
    for k, v in (hp or {}).items():
        if str(k).startswith("_"):
            continue
        full[k] = v
    return full


class NoAliasDumper(yaml.SafeDumper):
    """YAML dumper that always inlines repeated objects instead of emitting
    &idN / *idN anchors — those are baffling inside a human-readable
    metadata file ("what does *id001 mean?").  Use this everywhere we write
    metadata.yaml.
    """
    def ignore_aliases(self, data):
        return True


def dump_yaml(obj: Any, path: str) -> None:
    """Write ``obj`` to ``path`` with the no-alias, key-order-preserving dumper."""
    with open(path, "w") as f:
        yaml.dump(obj, f, Dumper=NoAliasDumper,
                  default_flow_style=False, allow_unicode=True, sort_keys=False)


# Keys (in order) that — when present in hp — get folded into the run
# folder name so every run's identity is visible at a glance instead of
# needing to open metadata.yaml.  Short prefixes keep paths readable.
_RUN_ID_HP_KEYS: list[tuple[str, str]] = [
    ("physics_weight",      "pw"),
    ("data_train_fraction", "frac"),
    ("learning_rate",       "lr"),
    ("weight_decay",        "wd"),
    ("dropout",             "do"),
    ("batch_size",          "bs"),
    ("hidden_layers",       "hl"),
]


def _fmt_hp_value(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "-".join(str(int(x)) if isinstance(x, (int, float)) and float(x).is_integer()
                        else str(x) for x in v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        # Short form: 0.001 → "1e-3", 3e-4 → "3e-4", 0.1 → "0.1"
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")
        if float(v).is_integer():
            return str(int(v))
        return f"{v:g}"
    return str(v)


def build_run_id(model_type: str, *, epochs_trained: int, rmse: float,
                 hp: dict | None = None, timestamp: str | None = None) -> str:
    """Build the run folder name.

    Layout: ``<Model>_ep<N>_rmse<R>_pw..._frac..._wd..._do..._lr...<_hl...>_<stamp>``.
    Missing keys are skipped.  Dataset fraction (``frac``) is always included
    when present since the data-efficiency study hinges on it.
    """
    parts = [model_type, f"ep{int(epochs_trained)}", f"rmse{float(rmse):.5f}"]
    hp = hp or {}
    for key, prefix in _RUN_ID_HP_KEYS:
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
    """Write both checkpoints with a uniform schema. Returns (best_path, final_path).

    Call this AFTER restoring the best weights into ``model`` — the model's
    current ``state_dict()`` is treated as the best snapshot; ``final_state``
    holds the weights observed at the last training epoch (captured before
    restoration).
    """
    os.makedirs(save_dir, exist_ok=True)

    best_path  = os.path.join(save_dir, BEST_CKPT_NAME)
    final_path = os.path.join(save_dir, FINAL_CKPT_NAME)

    common = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_class":    model_cls_name,
        "hparams":        hparams_blob,
        "norm_stats":     norm_stats,
        "epochs_trained": int(epochs_trained),
    }

    torch.save({
        **common,
        "model_state":      model.state_dict(),
        "checkpoint_kind":  "best",
        "best_epoch":       int(best_epoch),
        "metrics":          avg_metrics,
        "val_metrics":      val_metrics,
        "test_metrics":     test_metrics,
    }, best_path)

    torch.save({
        **common,
        "model_state":      final_state,
        "checkpoint_kind":  "final",
    }, final_path)

    return best_path, final_path
