"""Data loader for pre-processed robot trajectory datasets.

This module is a pure loader -- all filtering, differentiation, and physics
computation happens in preprocess_data.py (the GUI preprocessor) at build time.
The resulting CSVs are loaded directly here with no re-processing.

Responsibilities
----------------
1. load_raw_sample(json_path)     -- parse a single raw JSON hardware log
2. RobotDataset (PyTorch Dataset) -- load pre-materialised filtered CSVs
3. make_dataloaders(run_dir, ...) -- convenience wrapper for train/val/test

On-disk layout (created by preprocess_data.py)
----------------------------------------------
  run_dir/
    metadata.json                         -- full provenance and normalisation
    train/  val/  test/
      t.csv                                -- timestamps (N,)
      raw_q.csv  raw_qd.csv  raw_qdd.csv  -- raw kinematics  (N, 5)
      raw_tau_measured.csv                 -- raw measured torque
      raw_tau_decomposed.csv               -- RNEA per-component from raw kinematics (N, 20)
      filtered_q.csv  filtered_qd.csv  filtered_qdd.csv
      filtered_tau_measured.csv
      filtered_tau_decomposed.csv          -- [tau_g(5), tau_M(5), tau_C(5), tau_f(5)] (N, 20)

Training uses the ``filtered_*`` CSVs exclusively.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

ACTIVE_JOINTS = 5  # Kikobot arm (matches legacy robot_physics export)

logger = logging.getLogger(__name__)

# CSV file names (must match preprocess_data.py output)
CSV_T                        = "t.csv"
CSV_RAW_Q                    = "raw_q.csv"
CSV_RAW_QD                   = "raw_qd.csv"
CSV_RAW_QDD                  = "raw_qdd.csv"
CSV_RAW_TAU_MEASURED         = "raw_tau_measured.csv"
CSV_RAW_TAU_DECOMPOSED       = "raw_tau_decomposed.csv"
CSV_FILTERED_Q               = "filtered_q.csv"
CSV_FILTERED_QD              = "filtered_qd.csv"
CSV_FILTERED_QDD             = "filtered_qdd.csv"
CSV_FILTERED_TAU_MEASURED    = "filtered_tau_measured.csv"
CSV_FILTERED_TAU_DECOMPOSED  = "filtered_tau_decomposed.csv"

METADATA_FILE = "metadata.json"

NORMALISATION_KEYS = (
    "mean_q", "std_q", "mean_qd", "std_qd",
    "mean_qdd", "std_qdd", "mean_tau", "std_tau",
    "mean_tau_a", "std_tau_a",
)

JOINT_NAMES  = [f"J{i+1}" for i in range(ACTIVE_JOINTS)]
JOINT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# =============================================================================
# TRIM HELPER (used by preprocess_data.py)
# =============================================================================

def resolve_front_back_trim(
        total_samples: int,
        front_percent: float = 0.0,
        back_percent: float = 0.0,
        min_remaining: int = 10,
) -> tuple[int, int]:
    """
    Compute (front_n, back_n) -- number of samples to remove from each end.

    Both percentages are computed against the full untrimmed length.
    Raises ValueError if combined trim leaves fewer than ``min_remaining``.
    """
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    fp = max(0.0, float(front_percent or 0.0))
    bp = max(0.0, float(back_percent  or 0.0))
    front_n = int(round(total_samples * fp / 100.0)) if fp > 0 else 0
    back_n  = int(round(total_samples * bp / 100.0)) if bp > 0 else 0
    if front_n + back_n > total_samples - min_remaining:
        raise ValueError(
            f"Cannot trim {front_n}+{back_n} from {total_samples} samples -- "
            f"at least {min_remaining} must remain."
        )
    return front_n, back_n


# =============================================================================
# RAW JSON LOADING
# =============================================================================

def load_raw_sample(json_path: str) -> tuple[dict, dict, int]:
    """
    Parse a single hardware log JSON file (schema hwrl_execution_log_v4).

    Returns
    -------
    L : dict of NumPy arrays  (time-series data)
    M : dict of scalars/dicts (run metadata)
    N : number of log entries
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    log = data["log"]
    N   = len(log)
    if N == 0:
        raise ValueError(f"Empty log in {json_path}")

    L = {
        "t":       np.array([e["t"]       for e in log]),
        "wp":      np.array([e["wp"]      for e in log]),
        "ee_err":  np.array([e["ee_err"]  for e in log]),
        "cmd_pos": np.array([e["cmd_pos"] for e in log]),
        "cmd_vel": np.array([e["cmd_vel"] for e in log]),
        "cmd_acc": np.array([e["cmd_acc"] for e in log]),
        "act_pos": np.array([e["act_pos"] for e in log]),
        "act_vel": np.array([e["act_vel"] for e in log]),
        "load":    np.array([e["load"]    for e in log]),
        "voltage": np.array([e["voltage"] for e in log]),
        "current": np.array([e["current"] for e in log]),
        "cmd_ee":  np.array([e["cmd_ee"]  for e in log]),
        "act_ee":  np.array([e["act_ee"]  for e in log]),
    }

    M = {
        "run_id":        data["run_id"],
        "label":         data["label"],
        "success":       data["success"],
        "dof":           data["dof"],
        "servo_ids":     data["servo_ids"],
        "ticks_to_rad":  data["ticks_to_rad"],
        "ticks_per_rev": data["ticks_per_revolution"],
        "num_entries":   data["num_log_entries"],
        "ctrl_hz":       data["actual_control_hz"],
        "fb_hz":         data["actual_feedback_hz"],
        "duration":      data["actual_duration_sec"],
        "tracking":      data["tracking_quality"],
        "geometry":      data["geometry"],
        "joint_map":     data["joint_mapping"],
        "source_file":   os.path.basename(json_path),
    }

    return L, M, N


# =============================================================================
# METADATA LOADING
# =============================================================================

def _load_run_metadata(run_dir: str) -> dict:
    """Load the run-level metadata.json."""
    path = os.path.join(run_dir, METADATA_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"metadata.json not found in {run_dir}")
    with open(path, "r") as f:
        return json.load(f)


def get_processed_split_path(run_dir: str, split: str, must_exist: bool = False) -> str:
    """Return the path to a split directory (train/val/test)."""
    path = os.path.join(run_dir, split)
    if must_exist and not os.path.isdir(path):
        raise FileNotFoundError(f"Split directory not found: {path}")
    return path


def _load_csv(split_dir: str, filename: str) -> np.ndarray:
    """Load a CSV with header row, returning float32 array."""
    path = os.path.join(split_dir, filename)
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)


def _npy_name_from_csv(filename: str) -> str:
    return os.path.splitext(filename)[0] + ".npy"


def _load_csv_or_npy(
    split_dir: str, csv_filename: str, *, use_memmap: bool
) -> np.ndarray:
    """Load a column file: prefer ``.npy`` (optional memory-mapped) if present.

    When ``use_memmap`` is True and ``<stem>.npy`` exists next to the CSV, load
    with ``numpy.load(..., mmap_mode='r')`` to avoid holding the full array in
    resident RAM.  Otherwise fall back to CSV via ``_load_csv`` (in-memory
    float32).  See ``Neural_Networks.tools.convert_run_csv_to_npy``.
    """
    npy_path = os.path.join(split_dir, _npy_name_from_csv(csv_filename))
    if use_memmap and os.path.isfile(npy_path):
        return np.load(npy_path, mmap_mode="r")
    return _load_csv(split_dir, csv_filename)


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class RobotDataset(Dataset):
    """
    PyTorch Dataset loading pre-materialised filtered CSVs.

    Supports two modes:
      pointwise : each sample is one timestep  -> shape (15,)
      sequence  : each sample is a window of T timesteps -> shape (T, 15)

    Input features:  [q(5), qd(5), qdd(5)] = 15  (from filtered CSVs)
    Target:          tau_measured (5,)             (filtered, normalised with mean_tau/std_tau)
    Physics:         decomposed [tau_g, tau_M, tau_C, tau_f] (20,)  (normalised with mean_tau/4 and std_tau for sum consistency)
    """

    def __init__(self,
                 run_dir: str,
                 split: str = "train",
                 mode: str = "pointwise",
                 seq_len: int = 50,
                 stride: int = 1,
                 normalise: bool = True,
                 use_memmap: bool = False,
                 ):
        if mode not in ("pointwise", "sequence"):
            raise ValueError(f"mode must be 'pointwise' or 'sequence', got {mode!r}")

        split_dir = get_processed_split_path(run_dir, split, must_exist=True)
        meta = _load_run_metadata(run_dir)

        # Load filtered arrays (what the model trains on)
        self._memmap = bool(use_memmap)
        _load = lambda fn: _load_csv_or_npy(split_dir, fn, use_memmap=self._memmap)
        self.q   = _load(CSV_FILTERED_Q)
        self.qd  = _load(CSV_FILTERED_QD)
        self.qdd = _load(CSV_FILTERED_QDD)
        self.tau_measured   = _load(CSV_FILTERED_TAU_MEASURED)
        self.tau_analytical = _load(CSV_FILTERED_TAU_DECOMPOSED)

        if self.tau_analytical.shape[-1] != 4 * ACTIVE_JOINTS:
            raise ValueError(
                f"Expected 20-dim decomposed physics CSV in {split_dir}, "
                f"got shape {self.tau_analytical.shape}. Rebuild the dataset."
            )

        # Ensure 2D even for single-joint edge case
        if self.q.ndim == 1:
            self.q   = self.q.reshape(-1, 1)
            self.qd  = self.qd.reshape(-1, 1)
            self.qdd = self.qdd.reshape(-1, 1)
            self.tau_measured   = self.tau_measured.reshape(-1, 1)

        self.mode    = mode
        self.seq_len = seq_len
        self.stride  = stride
        self.run_dir = run_dir
        self.split   = split
        self.metadata = meta

        _backend = "memmap" if (self._memmap and any(
            os.path.isfile(os.path.join(split_dir, _npy_name_from_csv(n)))
            for n in (CSV_FILTERED_Q, CSV_FILTERED_QD)
        )) else "ram"
        logger.info("RobotDataset [%s]: loaded %d samples from %s  (%s)",
                    split, self.q.shape[0], split_dir, _backend)

        # Normalisation stats from training split (stored in metadata.json)
        norm = meta.get("normalisation", {})
        if normalise and norm:
            self.mean_q   = np.asarray(norm["mean_q"],   dtype=np.float32)
            self.std_q    = np.asarray(norm["std_q"],    dtype=np.float32).clip(min=1e-8)
            self.mean_qd  = np.asarray(norm["mean_qd"],  dtype=np.float32)
            self.std_qd   = np.asarray(norm["std_qd"],   dtype=np.float32).clip(min=1e-8)
            self.mean_qdd = np.asarray(norm["mean_qdd"], dtype=np.float32)
            self.std_qdd  = np.asarray(norm["std_qdd"],  dtype=np.float32).clip(min=1e-8)
            self.mean_tau = np.asarray(norm["mean_tau"], dtype=np.float32)
            self.std_tau  = np.asarray(norm["std_tau"],  dtype=np.float32).clip(min=1e-8)
            self.mean_tau_a = np.asarray(norm["mean_tau_a"], dtype=np.float32)
            self.std_tau_a  = np.asarray(norm["std_tau_a"],  dtype=np.float32).clip(min=1e-8)
        else:
            nj = ACTIVE_JOINTS
            _z = np.zeros(nj, dtype=np.float32)
            _o = np.ones(nj,  dtype=np.float32)
            self.mean_q = _z.copy(); self.std_q = _o.copy()
            self.mean_qd = _z.copy(); self.std_qd = _o.copy()
            self.mean_qdd = _z.copy(); self.std_qdd = _o.copy()
            self.mean_tau = _z.copy(); self.std_tau = _o.copy()
            self.mean_tau_a = _z.copy(); self.std_tau_a = _o.copy()

        # Sequence mode: build window start indices respecting trajectory boundaries
        if mode == "sequence":
            trajectories = meta.get("split", {}).get("stats", {}).get(
                split, {}).get("trajectories", [])
            if trajectories:
                self._starts = []
                for traj in trajectories:
                    t_start = traj["start_idx"]
                    t_end   = traj["end_idx_exclusive"]
                    for s in range(t_start, t_end - seq_len + 1, stride):
                        self._starts.append(s)
            else:
                N = self.q.shape[0]
                self._starts = list(range(0, N - seq_len + 1, stride))

    def _build_features(self, idx):
        """[q_norm, qd_norm, qdd_norm] concatenated."""
        q   = (self.q[idx]   - self.mean_q)   / self.std_q
        qd  = (self.qd[idx]  - self.mean_qd)  / self.std_qd
        qdd = (self.qdd[idx] - self.mean_qdd) / self.std_qdd
        return np.concatenate([q, qd, qdd], axis=-1).astype(np.float32)

    def _build_physics(self, idx):
        """Return normalised 20-dim decomposed physics: [tau_g(5), tau_M(5), tau_C(5), tau_f(5)].

        Each 5-dim block is normalised so that the SUM of all 4 blocks equals
        ``(tau_total - mean_tau) / std_tau`` — the exact same space as the target.

        Per-component: ``(comp - mean_tau/4) / std_tau``.  This distributes the
        target mean equally across the 4 physics components and uses the target
        std for scaling, ensuring mathematical consistency in every physics loss
        that sums normalised components (used when training models that consume
        analytical torque channels; ignored by BlackBoxFNN).
        """
        per_comp_mean = np.tile(self.mean_tau / 4.0, 4)  # (20,)
        per_comp_std  = np.tile(self.std_tau,         4)  # (20,)
        return ((self.tau_analytical[idx] - per_comp_mean) / per_comp_std).astype(np.float32)

    def __len__(self):
        if self.mode == "pointwise":
            return self.q.shape[0]
        return len(self._starts)

    def __getitem__(self, idx):
        if self.mode == "pointwise":
            features = self._build_features(idx)
            target   = ((self.tau_measured[idx] - self.mean_tau) / self.std_tau).astype(np.float32)
            physics  = self._build_physics(idx)
            return (
                torch.from_numpy(features),
                torch.from_numpy(target),
                torch.from_numpy(physics),
            )
        else:
            s = self._starts[idx]
            e = s + self.seq_len
            sl = slice(s, e)
            features = self._build_features(sl)
            target   = ((self.tau_measured[sl] - self.mean_tau) / self.std_tau).astype(np.float32)
            physics  = self._build_physics(sl)
            return (
                torch.from_numpy(features),
                torch.from_numpy(target),
                torch.from_numpy(physics),
            )

    @property
    def input_dim(self) -> int:
        return ACTIVE_JOINTS * 3  # [q, qd, qdd]

    @property
    def output_dim(self) -> int:
        return ACTIVE_JOINTS

    @property
    def physics_dim(self) -> int:
        return 4 * ACTIVE_JOINTS  # [tau_g, tau_M, tau_C, tau_f]


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

class _AttrPassthroughSubset(torch.utils.data.Subset):
    """Subset that forwards unknown attribute reads to the wrapped dataset.

    This lets downstream code keep using ``loaders['train'].dataset.mean_tau``
    etc. after we subsample — ``mean_tau``, ``std_tau``, ``metadata`` are all
    resolved on the underlying RobotDataset.
    """
    def __getattr__(self, name):
        # __getattr__ only runs when normal lookup fails — so indices/dataset
        # defined by Subset still take priority.
        # Use object.__getattribute__ for ``dataset`` so spawn unpickling never
        # re-enters __getattr__("dataset") (infinite recursion).
        _ds = object.__getattribute__(self, "dataset")
        return getattr(_ds, name)


def make_dataloaders(
        run_dir: str,
        batch_size: int = 256,
        mode: str = "pointwise",
        seq_len: int = 50,
        stride: int = 1,
        normalise: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 4,
        drop_last: bool = False,
        data_train_fraction: float = 1.0,
        data_train_seed: int = 0,
        use_memmap: bool = False,
) -> dict[str, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for train, val, test splits.

    ``data_train_fraction`` ∈ (0, 1] keeps only that fraction of the TRAIN
    samples (validation and test are never subsampled).  The subset is drawn
    with a deterministic RNG seeded by ``data_train_seed`` so runs with the
    same (fraction, seed) see the same samples — important for fair
    data-efficiency comparisons across models.

    ``use_memmap``: if ``.npy`` sidecars exist (see
    ``Neural_Networks.tools.convert_run_csv_to_npy``), load with memory-mapped
    numpy arrays to reduce process RSS.

    ``num_workers`` / ``prefetch_factor`` (when used from
    :func:`run_training` without ``NN_NUM_WORKERS``) are auto-capped on
    low-RAM machines to limit worker-process fan-out.

    Returns dict with 'train', 'val', 'test' DataLoaders.
    """
    if not (0.0 < float(data_train_fraction) <= 1.0):
        raise ValueError(f"data_train_fraction must be in (0, 1], got {data_train_fraction}")

    loaders = {}
    for split in ("train", "val", "test"):
        ds = RobotDataset(
            run_dir, split=split, mode=mode, seq_len=seq_len,
            stride=stride, normalise=normalise, use_memmap=use_memmap,
        )
        if split == "train" and float(data_train_fraction) < 1.0:
            import numpy as _np
            n_full = len(ds)
            n_keep = max(1, int(round(n_full * float(data_train_fraction))))
            rng    = _np.random.default_rng(int(data_train_seed))
            idx    = rng.permutation(n_full)[:n_keep]
            ds     = _AttrPassthroughSubset(ds, idx.tolist())
        # Only drop the last incomplete batch when the dataset is large enough
        # to guarantee at least 2 full batches — otherwise drop_last would
        # silently empty the DataLoader (0 batches, no training, no error).
        _eff_drop_last = drop_last and (split == "train") and (len(ds) >= 2 * batch_size)
        loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=_eff_drop_last,
        )
    return loaders
