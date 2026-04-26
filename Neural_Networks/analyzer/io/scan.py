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
    exist the most-recently-sorted one (by name, so run_MMDD_HHMM sorts
    chronologically) is chosen automatically.
    """
    root = Path(models_dir) if models_dir else _GRID_ROOT
    if _is_run_folder(root):
        return root
    runs = list_run_dirs(root)
    if not runs:
        raise FileNotFoundError(f"No run folders found under {root}")
    chosen = runs[-1]
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
    runs = list_run_dirs(_GRID_ROOT)
    if not runs:
        print(f"No run folders found under {_GRID_ROOT}")
        return 1
    print(f"\nAvailable run folders in {_GRID_ROOT}:\n")
    for i, d in enumerate(runs, 1):
        n = len(list(d.rglob("metadata.yaml")))
        m = re.match(r"^run_(\d{2})(\d{2})_", d.name)
        date = f"20xx-{m.group(1)}-{m.group(2)}" if m else "?"
        print(f"  [{i}] {d.name}  ({date}, {n} trials)")
    print(f"\nUsage: PYTHONPATH=. python -m Neural_Networks.analyzer --models-dir {runs[-1]}")
    print()
    return 0
