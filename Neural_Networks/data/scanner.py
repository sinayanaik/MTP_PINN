"""
Neural_Networks.data.scanner
==============================
Filesystem utilities for finding and loading processed training datasets.

A "processed dataset" is a directory under ``Neural_Networks/train_data/``
that contains ``metadata.json`` (or legacy ``metadata.yaml``) and the three
split sub-directories (train/, val/, test/).

No UI or Rich dependencies — all display is handled by tui/dataset_display.py.
"""

from __future__ import annotations

import json
import os

import yaml


def load_run_metadata(run_path: str) -> dict | None:
    """Load run-level metadata.json (v3) or metadata.yaml (legacy).

    Returns the parsed dict, or ``None`` if no metadata file exists or
    parsing fails.
    """
    # Prefer v3 JSON metadata
    json_path = os.path.join(run_path, "metadata.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    # Fall back to legacy YAML metadata
    yaml_path = os.path.join(run_path, "metadata.yaml")
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return None

    return None


def scan_existing_datasets(train_data_dir: str) -> list[dict]:
    """Scan *train_data_dir* for valid processed dataset directories.

    A directory is considered valid when it contains metadata and all three
    split sub-directories (train/, val/, test/).

    Returns
    -------
    list[dict]
        Each element is the metadata dict augmented with ``run_name`` and
        ``run_dir`` keys, sorted newest-first by mtime.
    """
    if not os.path.isdir(train_data_dir):
        return []

    valid: list[dict] = []
    for entry in sorted(
        os.scandir(train_data_dir),
        key=lambda e: e.stat().st_mtime,
        reverse=True,
    ):
        if not entry.is_dir():
            continue
        meta = load_run_metadata(entry.path)
        if meta is None:
            continue
        # Dataset is unusable unless all three split dirs exist
        splits_ok = all(
            os.path.isdir(os.path.join(entry.path, s))
            for s in ("train", "val", "test")
        )
        if not splits_ok:
            continue
        meta["run_name"] = entry.name
        meta["run_dir"]  = entry.path
        valid.append(meta)
    return valid
