"""Filesystem scan: find run folders and load metadata records."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from ..config import _ARCH_DIR_NAMES, _GRID_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run-folder detection
# ---------------------------------------------------------------------------

def _is_run_folder(p: Path) -> bool:
    if (p / "models_registry.yaml").is_file():
        return True
    return any((p / name).is_dir() for name in _ARCH_DIR_NAMES)


def list_run_dirs(root: Path) -> list[Path]:
    """All subdirs of root that look like a training run."""
    runs = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("_") and d.name != "analysis":
            if any(d.rglob("metadata.yaml")):
                runs.append(d)
    return runs


_list_run_dirs = list_run_dirs  # backward compat


def auto_select_run(models_dir: str | None = None) -> Path:
    """Return the most-recent run folder, auto-selecting if there is only one.

    Pass *models_dir* to override the default grid root.  If multiple runs
    exist the most-recently-sorted one (by name) is chosen automatically.
    Supports two-level hierarchy (Version/Run).
    """
    root = Path(models_dir) if models_dir else _GRID_ROOT
    
    # If the provided path is already a run folder, use it.
    if _is_run_folder(root):
        return root
        
    # Otherwise, list subdirs (could be versions or runs).
    candidates = list_run_dirs(root)
    if not candidates:
        raise FileNotFoundError(f"No run folders found under {root}")
        
    chosen = candidates[-1]
    
    # If chosen is not a run folder but contains sub-runs, go one level deeper.
    if not _is_run_folder(chosen):
        sub_runs = list_run_dirs(chosen)
        if sub_runs:
            chosen = sub_runs[-1]
            
    logger.info("Auto-selected run folder: %s", chosen.name)
    return chosen


def resolve_models_dir(models_dir: str) -> str:
    """Resolve to a single run-folder path (used by the CLI runner)."""
    return str(auto_select_run(models_dir))


# ---------------------------------------------------------------------------
# Record scanning
# ---------------------------------------------------------------------------

def scan_trained_models(models_dir: str) -> list[dict[str, Any]]:
    root = Path(models_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Trained-models directory not found: {models_dir}")

    records: list[dict[str, Any]] = []
    for meta_path in sorted(root.rglob("metadata.yaml")):
        try:
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
        except Exception as exc:
            logger.warning("Could not read %s: %s", meta_path, exc)
            continue
        if not isinstance(meta, dict):
            continue

        record: dict[str, Any] = dict(meta)
        record["_meta_path"] = str(meta_path)
        record["_run_dir"] = str(meta_path.parent)
        hist_path = meta_path.parent / "training_history.csv"
        record["_history_path"] = str(hist_path) if hist_path.is_file() else None
        records.append(record)

    if not records:
        logger.warning("No metadata.yaml files found under %s.", models_dir)
    return records


def group_by_model_type(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        mtype = rec.get("model_type") or Path(rec["_run_dir"]).parent.name
        groups[str(mtype)].append(rec)
    return dict(groups)


# ---------------------------------------------------------------------------
# CLI helper (used only by cli.py --list-datasets)
# ---------------------------------------------------------------------------

def list_datasets() -> int:
    """List all versions and their runs for the CLI."""
    versions = [d for d in sorted(_GRID_ROOT.iterdir()) if d.is_dir() and not d.name.startswith("_")]
    if not versions:
        print(f"No version folders found under {_GRID_ROOT}")
        return 1

    print(f"\nAvailable grid search versions in {_GRID_ROOT.name}/:\n")
    for i, v_dir in enumerate(versions, 1):
        runs = list_run_dirs(v_dir)
        print(f"  [{i}] {v_dir.name}  ({len(runs)} run folders)")
        for r_dir in runs:
            n = len(list(r_dir.rglob("metadata.yaml")))
            print(f"        - {r_dir.name}  ({n} trials)")
            
    last_run = None
    if versions:
        last_runs = list_run_dirs(versions[-1])
        if last_runs:
            last_run = last_runs[-1]
        else:
            last_run = versions[-1]

    if last_run:
        print(f"\nUsage: python3 -m Neural_Networks.analyzer --models-dir {last_run}")
    print()
    return 0
