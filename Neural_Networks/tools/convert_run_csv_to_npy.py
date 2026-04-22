#!/usr/bin/env python3
"""Convert preprocessed run CSVs to float32 ``.npy`` files for memory-mapped loading.

The training :class:`Neural_Networks.loader.RobotDataset` can read these with
``use_memmap=True`` (or ``NN_DATASET_MEMMAP=1`` in the environment) so large
splits are not fully resident in RAM.

Usage (from repo root)::

    PYTHONPATH=. python3 -m Neural_Networks.tools.convert_run_csv_to_npy \\
        Neural_Networks/train_data/run_xxx
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Re-use the same file names as loader.py
from Neural_Networks.loader import (  # noqa: E402
    CSV_FILTERED_Q,
    CSV_FILTERED_QD,
    CSV_FILTERED_QDD,
    CSV_FILTERED_TAU_DECOMPOSED,
    CSV_FILTERED_TAU_MEASURED,
    get_processed_split_path,
)


def _load_csv_raw(split_dir: str, filename: str) -> np.ndarray:
    path = os.path.join(split_dir, filename)
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)


def _csv_to_npy(split_dir: str, csv_name: str) -> None:
    base = os.path.splitext(csv_name)[0] + ".npy"
    out = os.path.join(split_dir, base)
    arr = _load_csv_raw(split_dir, csv_name)
    np.save(out, arr)  # standard uncompressed .npy
    print(f"  wrote {out}  shape={arr.shape}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", help="Path to a preprocess run (metadata.json + train/val/test/)")
    a = p.parse_args()
    run_dir = os.path.abspath(a.run_dir)
    for sp in ("train", "val", "test"):
        sp_dir = get_processed_split_path(run_dir, sp, must_exist=True)
        print(f"{sp}:")
        for fn in (
            CSV_FILTERED_Q,
            CSV_FILTERED_QD,
            CSV_FILTERED_QDD,
            CSV_FILTERED_TAU_MEASURED,
            CSV_FILTERED_TAU_DECOMPOSED,
        ):
            _csv_to_npy(sp_dir, fn)
    print("Done. Enable memmap in training: NN_DATASET_MEMMAP=1 or hyperparams['dataset_memmap']=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
