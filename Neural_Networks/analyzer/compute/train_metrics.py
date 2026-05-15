"""Per-record train-split inference + metric computation, with on-disk cache.

The cache lives at <run_dir>/train_metrics_cache.yaml. Bumping
TRAIN_METRICS_CACHE_VERSION (in config.py) invalidates all caches.

Torch is imported lazily inside `compute()` so callers in cache-only mode
never touch torch and stay torch-free.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config import N_JOINTS, TRAIN_METRICS_CACHE_VERSION

logger = logging.getLogger(__name__)


def _cache_path(rec: dict[str, Any]) -> Path:
    return Path(rec.get("_run_dir", "")) / "train_metrics_cache.yaml"


def has_cache(rec: dict[str, Any]) -> bool:
    """True iff a current-version cache file exists."""
    p = _cache_path(rec)
    if not p.is_file():
        return False
    try:
        with open(p) as f:
            cached = yaml.safe_load(f)
        return isinstance(cached, dict) and cached.get("_v") == TRAIN_METRICS_CACHE_VERSION
    except Exception:
        return False


def read_cache(rec: dict[str, Any]) -> dict[str, Any]:
    """Return cached metrics dict or {} if missing/stale."""
    p = _cache_path(rec)
    if not p.is_file():
        return {}
    try:
        with open(p) as f:
            cached = yaml.safe_load(f)
        if isinstance(cached, dict) and cached.get("_v") == TRAIN_METRICS_CACHE_VERSION:
            return {k: v for k, v in cached.items() if k != "_v"}
    except Exception:
        return {}
    return {}


_DATASET_CACHE: dict[Path, Any] = {}

def compute(rec: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Load model.pt, run inference on training split, compute full metrics.

    Caches result to <run_dir>/train_metrics_cache.yaml. When force=False
    and a current-version cache exists, returns it without touching torch.
    """
    if not force:
        cached = read_cache(rec)
        if cached:
            return cached

    run_dir = Path(rec.get("_run_dir", ""))
    cache_path = run_dir / "train_metrics_cache.yaml"

    model_pt = run_dir / "model.pt"
    if not model_pt.is_file():
        logger.debug("No model.pt in %s — skipping train metrics.", run_dir)
        return {}

    data_run_dir_raw = rec.get("data_run_dir", "")
    if not data_run_dir_raw:
        logger.debug("data_run_dir missing for %s — skipping.", run_dir)
        return {}
    
    data_run_dir = Path(data_run_dir_raw)
    if not data_run_dir.is_dir():
        # Try resolving relative to repo root if absolute path fails
        try:
            from ..config import _NN_ROOT
            repo_root = _NN_ROOT.parent # analyzer is in Neural_Networks/analyzer
            # Common pattern: .../Neural_Networks/train_data/run_name
            parts = data_run_dir.parts
            if "Neural_Networks" in parts:
                idx = parts.index("Neural_Networks")
                rel_path = Path(*parts[idx:])
                potential = repo_root / rel_path
                if potential.is_dir():
                    data_run_dir = potential
                else:
                    logger.debug("Could not resolve %s relative to %s", rel_path, repo_root)
                    return {}
            else:
                # Just try the name of the folder inside Neural_Networks/train_data
                potential = repo_root / "Neural_Networks" / "train_data" / data_run_dir.name
                if potential.is_dir():
                    data_run_dir = potential
                else:
                    logger.debug("Path %s invalid and no fallback found.", data_run_dir)
                    return {}
        except Exception as e:
            logger.debug("Fallback resolution failed: %s", e)
            return {}

    try:
        import torch
        from Neural_Networks.loader import RobotDataset
        from Neural_Networks.models.torque_models import (
            BlackBoxFNN, PhysicsRegularizedFNN, ResidualCorrectionFNN,
        )
        from Neural_Networks.models.shared.metrics_numpy import compute_metrics
    except ImportError as exc:
        logger.debug("Cannot import torch/model modules: %s", exc)
        return {}

    try:
        mtype = rec.get("model_type", "BlackBoxFNN")
        hp = rec.get("hyperparams", {})
        hidden_layers = hp.get("hidden_layers", [256, 512, 256])
        dropout       = float(hp.get("dropout", 0.1))
        activation    = str(hp.get("activation", "gelu"))

        _cls_map = {
            "BlackBoxFNN":           BlackBoxFNN,
            "PhysicsRegularizedFNN": PhysicsRegularizedFNN,
            "ResidualCorrectionFNN": ResidualCorrectionFNN,
        }

        ckpt = torch.load(str(model_pt), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            logger.debug("Checkpoint at %s is not a dict — skipping.", model_pt)
            return {}
        model_state = ckpt.get("model_state")
        if model_state is None:
            logger.debug(
                "Key 'model_state' not found in %s (found keys: %s) — skipping.",
                model_pt, list(ckpt.keys())[:8],
            )
            return {}

        ckpt_hp   = ckpt.get("hparams") or hp
        ckpt_hl   = ckpt_hp.get("hidden_layers", hidden_layers)
        ckpt_do   = float(ckpt_hp.get("dropout", dropout))
        ckpt_act  = str(ckpt_hp.get("activation", activation))
        cls_final = _cls_map.get(ckpt.get("model_class") or mtype)
        if cls_final is None:
            cls_final = _cls_map.get(mtype)
        if cls_final is None:
            return {}

        model = cls_final(n_joints=N_JOINTS, hidden_layers=ckpt_hl,
                          dropout=ckpt_do, activation=ckpt_act)
        
        # Handle torch.compile prefix that might be present in the saved state dict
        clean_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
        model.load_state_dict(clean_state)
        model.eval()

        global _DATASET_CACHE
        if data_run_dir not in _DATASET_CACHE:
            _DATASET_CACHE[data_run_dir] = RobotDataset(data_run_dir, split="train", mode="pointwise", normalise=True)
        dataset = _DATASET_CACHE[data_run_dir]
        
        ckpt_norm = ckpt.get("norm_stats", {})
        if ckpt_norm and "mean_tau" in ckpt_norm:
            mean_tau = np.asarray(ckpt_norm["mean_tau"], dtype=np.float32)
            std_tau  = np.asarray(ckpt_norm["std_tau"],  dtype=np.float32).clip(min=1e-8)
        else:
            mean_tau = dataset.mean_tau
            std_tau  = dataset.std_tau

        all_preds: list[np.ndarray] = []
        all_tgts:  list[np.ndarray] = []
        batch_size = 2048
        n = len(dataset)

        with torch.no_grad():
            f_all, t_all, p_all = dataset[slice(0, n)]
            
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                feat = f_all[start:end]
                tgt  = t_all[start:end]
                phy  = p_all[start:end]
                
                pred_norm = model(feat, phy)
                pred_phys = pred_norm.numpy() * std_tau + mean_tau
                tgt_phys  = tgt.numpy()       * std_tau + mean_tau
                all_preds.append(pred_phys)
                all_tgts.append(tgt_phys)

        pred_np = np.concatenate(all_preds, axis=0)
        tgt_np  = np.concatenate(all_tgts,  axis=0)

        result: dict[str, Any] = dict(compute_metrics(pred_np, tgt_np))

        try:
            with open(cache_path, "w") as f:
                yaml.dump({"_v": TRAIN_METRICS_CACHE_VERSION, **result}, f)
        except Exception:
            pass

        return result

    except Exception as exc:
        logger.debug("Train metrics computation failed for %s: %s", run_dir, exc)
        return {}
