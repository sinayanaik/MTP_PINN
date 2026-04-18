"""Grid-search runner for KikoBot inverse-dynamics models.

**Training boundary:** this module calls ``Neural_Networks.core.trainer.train_model``
only (parallel or sequential workers).  It does **not** import
``Neural_Networks.apps.train`` — the interactive Rich CLI is a separate entry
point.  Both share low-level helpers (e.g. ``train_epoch``) via ``core``.

Sweeps a Cartesian grid of hyperparameters across one or more models and
saves every cell into an isolated, timestamped directory so the main
``Trained_Models`` registry stays clean and each invocation of the grid
search lives in its own folder.

Grid config: plain Python module under ``Neural_Networks/config/grids/``.
Each module exposes the following module-level names::

    STUDY_NAME       : str          # human-readable study tag
    SEEDS_PER_CELL   : int          # replicas per (cell × model)
    RUN_DIR          : str | None   # None → newest preprocessed dataset
    BASE             : dict         # fixed hp every cell
    AXES             : dict[str, list]   # Cartesian product axes
    MODELS           : list[str]

``STUDY_NAME`` and sweep axes come from the **loaded preset** (see below).
Parallel worker count is chosen from VRAM and CPU tier.

**Preset selection (in order)**

1. **Optional** ``NN_GRID_PRESET=hpc`` or ``laptop`` — forces that preset
   (overrides auto-detection).  Run-time knobs still come from ``active.py``.
2. **Auto HPC** — if CUDA is available, GPU VRAM is at least 40 GB, and
   either ``detect_hardware()["profile"] == "server"`` **or** a common batch
   scheduler env var is set (``SLURM_JOB_ID``, ``PBS_JOBID``, ``LSB_JOBID``,
   ``SGE_CLUSTER_NAME``, ``JOB_ID``), load ``config/grids/hpc.py`` with no
   exports required.
3. Otherwise load ``Neural_Networks.config.grids.active`` (edit its import
   line for local laptop vs hpc defaults).

At startup the chosen preset is printed (e.g. ``preset: hpc (auto: ...)``).

Output layout::

    Neural_Networks/Trained_Models_GridSearch/
        batch_<NNN>_<Nperms>perms_<timestamp>/   # one folder per invocation
            metadata.json             # extensible run metadata
            grid_config.py            # copy of the config module used
            grid_log.jsonl            # one line per completed trial
            <ModelName>/<run_id>/…    # per-run artefacts (grouped by model)

Axes that don't affect a given model (``physics_weight`` for non-α-active,
``hidden_layers`` for structured) are projected out automatically.

Resume an existing study-run:
    python -m Neural_Networks.apps.grid_search --grid laptop --resume <timestamp_dir>

Usage::

    python -m Neural_Networks.apps.grid_search

Edit ``Neural_Networks/config/grids/active.py`` for local defaults; large
GPU / batch-job nodes pick the HPC sweep automatically as described above.

**Parallelism tuning (optional env)**

All are optional; unset values use profile-based defaults.

- ``NN_GRID_PER_MODEL_GB`` — float, GB assumed per concurrent training process
  (default: 4.5 server / 5.5 desktop / 6.0 laptop, cpu, or other).
- ``NN_GRID_CUDA_CTX_GB`` — float, GPU memory headroom for CUDA contexts (default 2.0).
- ``NN_GRID_CPU_RESERVE`` — int, CPU cores to leave free (default: 1 on server,
  2 otherwise).  ``cpu_limit = max(1, cpu_count - reserve)``.
- ``NN_GRID_MAX_WORKERS`` / ``NN_GRID_MIN_WORKERS`` — hard cap / floor on
  parallel trial count after VRAM and CPU limits.
- ``NN_GRID_SINGLE_GPU`` — if ``1``/``true``/``yes``, treat only the first
  visible CUDA device for VRAM planning (default: use all visible GPUs).
- ``NN_GRID_STATUS_INTERVAL`` — seconds between parallel-run status refreshes
  on the parent (default **2** on a TTY, **15** otherwise; first tick ~2s).
  On a TTY the summary is redrawn in place (no scrolling); otherwise one
  short block per tick.
- ``NN_GRID_MAX_TRIALS_PER_GPU`` — if set to a positive integer, caps
  ``n_workers`` at this value times the number of visible GPUs (reduces
  per-GPU oversubscription; often **improves** throughput and utilisation).
- ``NN_GRID_DL_WORKERS`` — dataloader ``num_workers`` for each parallel
  trial process (default: auto up to **2** when ``n_workers > 1``; was **0**).
- ``NN_GRID_COMPILE_MEM_FRAC`` — disable ``torch.compile`` in a worker only
  when ``gpu_memory_fraction`` is below this (default **0.12**; parallel runs
  used to force-compile off for almost all ``mem_fraction`` values).
- ``NN_GRID_RICH_TUI`` — set to ``1``/``true``/``yes`` on a TTY to use a
  ``rich.live.Live`` table (per-worker rows) instead of the ASCII panel.
- ``NN_GRID_MODE`` — ``trial`` or ``serious``: selects sweep size (see preset
  ``BASE_TRIAL`` / ``AXES_TRIAL`` vs full ``BASE`` / ``AXES``).  Overrides the
  TTY prompt when set.  Non-TTY defaults to **serious**.
- ``NN_GRID_PROGRESS_EVERY`` — in each worker, update shared epoch status
  every N training epochs (default 1; raise on huge grids to reduce IPC).
- ``NN_GRID_ETA_SEED_CAP_S`` — upper bound (seconds) on per-trial duration when
  ETA is seeded from epoch-1 wall time only (default 7200).
- ``NN_GRID_ETA_SEED_K`` — effective epoch count for that seed:
  ``median(epoch1_wall) × K`` is blended with the naive ``epoch1 × epochs_max``
  guess (default K=96).

When multiple CUDA devices are visible, parallel grid search scales the
VRAM-based worker budget by device count and assigns trials round-robin
across GPUs (``torch.cuda.set_device`` in each trial).  Set
``NN_GRID_SINGLE_GPU=1`` to ignore extra GPUs and pack ``cuda:0`` only.
For sharing one GPU across jobs, see NVIDIA Multi-Process Service (MPS).
"""
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
import fcntl
import hashlib
import importlib
import importlib.util
import itertools
import json
import multiprocessing
import os
import platform
import random
import shutil
import statistics
import socket
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import torch

from Neural_Networks.core.trainer import train_model
from Neural_Networks.cli.hp_wizard import get_default_hp
from Neural_Networks.config.hardware import detect_hardware
from Neural_Networks.data.scanner import scan_existing_datasets
from Neural_Networks.models import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_APPS_DIR       = os.path.dirname(os.path.abspath(__file__))
_NN_DIR         = os.path.dirname(_APPS_DIR)
TRAIN_DATA_DIR  = os.path.join(_NN_DIR, "train_data")
GRID_CONFIG_DIR = os.path.join(_NN_DIR, "config", "grids")
GRID_OUTPUT_DIR = os.path.join(_NN_DIR, "Trained_Models_GridSearch")


# ---------------------------------------------------------------------------
# Auto-detect parallel workers based on available VRAM
# ---------------------------------------------------------------------------

_DEFAULT_PER_MODEL_GB_SERVER  = 4.5   # tighter packing on A100/H100-class nodes
_DEFAULT_PER_MODEL_GB_DESKTOP = 5.5
_DEFAULT_PER_MODEL_GB_OTHER   = 6.0   # laptop / cpu fallback
_CUDA_CTX_GB_DEFAULT = 2.0            # per-GPU context headroom (GiB scale)

_SCHEDULER_ENV_KEYS = (
    "SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "SGE_CLUSTER_NAME", "JOB_ID",
)


def _has_batch_scheduler() -> bool:
    """True if the process appears to run under a batch workload manager."""
    return any(os.environ.get(k, "").strip() for k in _SCHEDULER_ENV_KEYS)


def _auto_grid_preset_name() -> str | None:
    """Return ``hpc`` when the node looks like a large-GPU cluster worker.

    Requires CUDA, VRAM >= 40 GB, and either the ``server`` hardware profile
    or a known scheduler environment variable (same VRAM gate).
    """
    if not torch.cuda.is_available():
        return None
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb < 40.0:
        return None
    hw = detect_hardware()
    if hw.get("profile") == "server":
        return "hpc"
    if _has_batch_scheduler():
        return "hpc"
    return None


def _per_model_gb_for_profile(profile: str) -> float:
    """VRAM budget (GiB) assumed per concurrent training process."""
    if profile == "server":
        return _DEFAULT_PER_MODEL_GB_SERVER
    if profile == "desktop":
        return _DEFAULT_PER_MODEL_GB_DESKTOP
    return _DEFAULT_PER_MODEL_GB_OTHER


def _cuda_ctx_gb() -> float:
    raw = os.environ.get("NN_GRID_CUDA_CTX_GB", "").strip()
    if not raw:
        return float(_CUDA_CTX_GB_DEFAULT)
    try:
        return max(0.1, float(raw))
    except ValueError:
        return float(_CUDA_CTX_GB_DEFAULT)


def _parallel_worker_plan() -> dict[str, Any]:
    """Compute VRAM/CPU-based parallel trial count and diagnostics for metadata."""
    env_used: list[str] = []
    cuda_ctx = _cuda_ctx_gb()
    if os.environ.get("NN_GRID_CUDA_CTX_GB", "").strip():
        env_used.append("NN_GRID_CUDA_CTX_GB")

    if not torch.cuda.is_available():
        return {
            "n_workers":           1,
            "vram_gb":             0.0,
            "profile":             "no_cuda",
            "per_model_gb":        0.0,
            "cuda_ctx_gb":         cuda_ctx,
            "n_gpus":              0,
            "n_from_vram":         1,
            "n_from_vram_per_gpu": 1,
            "cpu_limit":           1,
            "cpu_reserve":         0,
            "cpu_count":           os.cpu_count() or 0,
            "env_overrides":       env_used,
        }

    n_gpus = max(1, int(torch.cuda.device_count()))
    _sg = os.environ.get("NN_GRID_SINGLE_GPU", "").strip().lower()
    if _sg in ("1", "true", "yes"):
        n_gpus = 1
        env_used.append("NN_GRID_SINGLE_GPU")

    vram_gb = min(
        torch.cuda.get_device_properties(i).total_memory / 1e9
        for i in range(n_gpus)
    )
    try:
        profile = str(detect_hardware().get("profile", "cpu"))
    except Exception:
        profile = "cpu"

    pmg_base = _per_model_gb_for_profile(profile)
    per_model_gb = pmg_base
    raw_pmg = os.environ.get("NN_GRID_PER_MODEL_GB", "").strip()
    if raw_pmg:
        try:
            per_model_gb = max(0.5, float(raw_pmg))
            env_used.append("NN_GRID_PER_MODEL_GB")
        except ValueError:
            per_model_gb = pmg_base

    usable = max(0.0, vram_gb - cuda_ctx)
    n_from_vram_per_gpu = max(1, int(usable / max(per_model_gb, 0.01)))
    n_from_vram = max(1, n_from_vram_per_gpu * n_gpus)

    cr_raw = os.environ.get("NN_GRID_CPU_RESERVE", "").strip()
    if cr_raw:
        try:
            cpu_reserve = max(0, int(cr_raw, 10))
            env_used.append("NN_GRID_CPU_RESERVE")
        except ValueError:
            cpu_reserve = 1 if profile == "server" else 2
    else:
        cpu_reserve = 1 if profile == "server" else 2

    ncpu = os.cpu_count() or 4
    cpu_limit = max(1, ncpu - cpu_reserve)

    n = min(n_from_vram, cpu_limit)
    return {
        "n_workers":           int(n),
        "vram_gb":             float(vram_gb),
        "profile":             profile,
        "per_model_gb":        float(per_model_gb),
        "cuda_ctx_gb":         float(cuda_ctx),
        "n_gpus":              int(n_gpus),
        "n_from_vram":         int(n_from_vram),
        "n_from_vram_per_gpu": int(n_from_vram_per_gpu),
        "cpu_limit":           int(cpu_limit),
        "cpu_reserve":         int(cpu_reserve),
        "cpu_count":           int(ncpu),
        "env_overrides":       list(env_used),
    }


def _apply_env_worker_bounds(n_workers: int) -> tuple[int, list[str]]:
    """Apply ``NN_GRID_MAX_WORKERS`` / ``NN_GRID_MIN_WORKERS``; return (n, env_used)."""
    env_used: list[str] = []
    n = int(max(1, n_workers))
    hi: int | None = None
    lo = 1
    raw_max = os.environ.get("NN_GRID_MAX_WORKERS", "").strip()
    if raw_max.isdigit():
        hi = max(1, int(raw_max))
        env_used.append("NN_GRID_MAX_WORKERS")
    raw_min = os.environ.get("NN_GRID_MIN_WORKERS", "").strip()
    if raw_min.isdigit():
        lo = max(1, int(raw_min))
        env_used.append("NN_GRID_MIN_WORKERS")
    if hi is not None:
        lo = min(lo, hi)
    upper = hi if hi is not None else 10**9
    n = max(lo, min(n, upper))
    return max(1, n), env_used


def _auto_max_workers() -> int:
    """Return the maximum number of parallel training workers (VRAM + CPU)."""
    plan = _parallel_worker_plan()
    n, _ = _apply_env_worker_bounds(plan["n_workers"])
    return int(n)


# ---------------------------------------------------------------------------
# Axis relevance — a model only actually uses certain axes; we skip cells
# that differ only on irrelevant axes to avoid wasted compute.
# ---------------------------------------------------------------------------

# Convex α data/physics mixture in ``trainer._compute_loss`` — only these
# models change the loss when ``physics_weight`` changes.  (Broader
# ``PHYSICS_WEIGHT_MODELS`` in ``Neural_Networks.models`` also lists structured
# PINNs for HP docs, but their losses do not use that α sweep.)
_ALPHA_ACTIVE_MODELS = {"PhysicsRegularizedFNN", "EquationConstrainedPINNFNN"}
_STRUCTURED_MODELS   = {"LagrangianStructuredFNN", "DecomposedStructuredPINNFNN"}


def relevant_axes_for(model_type: str, axes: dict) -> list[str]:
    """Return axis names that actually change the loss for this model."""
    keys = list(axes.keys())
    if model_type not in _ALPHA_ACTIVE_MODELS and "physics_weight" in keys:
        keys.remove("physics_weight")
    if model_type in _STRUCTURED_MODELS and "hidden_layers" in keys:
        keys.remove("hidden_layers")
    return keys


# ---------------------------------------------------------------------------
# Cell ID — deterministic short hash of (model, axis values)
# ---------------------------------------------------------------------------

def cell_id(model_type: str, cell_hp: dict, relevant: list[str]) -> str:
    payload = {"model": model_type, **{k: cell_hp[k] for k in sorted(relevant)}}
    j = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(j.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------

def iter_cells(axes: dict[str, list]) -> Iterable[dict]:
    if not axes:
        yield {}
        return
    keys   = list(axes.keys())
    values = [axes[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def build_trial_list(
    models: list[str],
    axes: dict[str, list],
    seeds: int,
) -> list[dict]:
    trials: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    for model in models:
        rel = relevant_axes_for(model, axes)
        reduced_axes = {k: axes[k] for k in rel}
        default_fill = {k: axes[k][0] for k in axes if k not in rel}
        for cell in iter_cells(reduced_axes):
            full_cell = {**default_fill, **cell}
            cid = cell_id(model, full_cell, rel)
            for s in range(seeds):
                key = (model, cid, s)
                if key in seen:
                    continue
                seen.add(key)
                trials.append({
                    "model":    model,
                    "cell_id":  cid,
                    "cell_hp":  full_cell,
                    "relevant": rel,
                    "seed":     s,
                })
    return trials


def _grid_combination_breakdown(
    models: list[str], axes: dict[str, list], seeds: int,
) -> list[dict[str, Any]]:
    """Per-model cell counts (Cartesian product over relevant axes) × seeds.

    Matches the expansion logic in ``build_trial_list`` (relevant-axis
    projection only; deduplication of ``(model, cell_id, seed)`` is rare).
    """
    rows: list[dict[str, Any]] = []
    for model in models:
        rel = relevant_axes_for(model, axes)
        dropped = [k for k in axes if k not in rel]
        if not rel:
            n_cells = 1
            factor_labels: list[str] = []
            axis_lens: list[int] = []
        else:
            n_cells = 1
            factor_labels = []
            axis_lens = []
            for k in rel:
                nk = len(axes[k])
                n_cells *= nk
                axis_lens.append(nk)
                factor_labels.append(f"{k}({nk})")
        rows.append({
            "model":          model,
            "relevant_axes":  list(rel),
            "dropped_axes":   dropped,
            "axis_factors":   factor_labels,
            "axis_lens":      axis_lens,
            "n_cells":        int(n_cells),
            "seeds_per_cell": int(seeds),
            "n_trials":       int(n_cells * seeds),
        })
    return rows


def _combinations_summary(
    models: list[str], axes: dict[str, list], seeds: int, trials: list[dict],
) -> dict[str, Any]:
    """Structured combination counts for metadata and console reporting."""
    breakdown = _grid_combination_breakdown(models, axes, seeds)
    by_model = Counter(t["model"] for t in trials)
    theory_total = sum(r["n_trials"] for r in breakdown)
    actual_total = len(trials)
    return {
        "seeds_per_cell": int(seeds),
        "per_model": breakdown,
        "total_theoretical_trials": theory_total,
        "total_expanded_trials": actual_total,
        "per_model_expanded_counts": dict(by_model),
    }


def _combination_report_width() -> int:
    """Target wrap width for combination detail text (TTY-aware)."""
    try:
        c = int(shutil.get_terminal_size(fallback=(100, 24)).columns)
    except Exception:
        c = 100
    return max(72, min(c, 120))


def _print_combination_report(summary: dict[str, Any]) -> None:
    """Print ``_combinations_summary``: numeric summary table + wrapped detail."""
    seeds = int(summary["seeds_per_cell"])
    breakdown: list[dict[str, Any]] = summary["per_model"]
    by_model: dict[str, int] = summary["per_model_expanded_counts"]
    theory_total = int(summary["total_theoretical_trials"])
    actual_total = int(summary["total_expanded_trials"])
    tw = _combination_report_width()
    ind = "    "
    ind2 = "      "

    print(f"  combinations  (seeds_per_cell = {seeds})")
    print()
    # --- Numeric summary (aligned; model column may truncate very long names)
    w_name = max(28, min(tw - 36, 48))
    print(ind + "Summary")
    print(ind + "-" * min(tw - len(ind), w_name + 32))
    hdr = (
        f"{'model':<{w_name}}  "
        f"{'cells':>7}  {'seeds':>7}  {'trials':>8}  {'list':>6}"
    )
    print(ind + hdr)
    print(ind + "-" * min(tw - len(ind), w_name + 32))
    for r in breakdown:
        m = str(r["model"])
        if len(m) > w_name:
            m = m[: max(1, w_name - 1)] + "…"
        nc = int(r["n_cells"])
        nt = int(r["n_trials"])
        raw_name = str(r["model"])
        actual = int(by_model.get(raw_name, 0))
        print(
            ind + f"{m:<{w_name}}  "
            f"{nc:>7d}  {seeds:>7d}  {nt:>8d}  {actual:>6d}",
        )
    print(ind + "-" * min(tw - len(ind), w_name + 32))
    parts_sum = "+".join(str(int(r["n_trials"])) for r in breakdown)
    print(
        ind + f"Σ trials  {parts_sum}  =  {theory_total}     "
        f"(expanded list length = {actual_total})",
    )
    if theory_total != actual_total:
        note = (
            f"Σ trials ({theory_total}) ≠ list length ({actual_total}) — "
            "duplicate (model, cell_id, seed) keys were merged."
        )
        print(textwrap.fill(note, width=tw, initial_indent=ind, subsequent_indent=ind))
    print()

    # --- Full strings with wrapping (nothing important truncated)
    print(ind + "Detail  (axis names and omitted axes; lines wrap to terminal width)")
    print(ind + "-" * min(tw - len(ind), tw - 4))
    lab_w = 22
    lab_omit = f"{'not swept:':<{lab_w}}"
    lab_lat = f"{'relevant axes:':<{lab_w}}"
    lab_sizes = f"{'sizes only:':<{lab_w}}"
    cont = ind2 + " " * lab_w
    for r in breakdown:
        m = str(r["model"])
        dropped: list[str] = list(r.get("dropped_axes") or [])
        factors: list[str] = list(r.get("axis_factors") or [])
        lens: list[int] = list(r.get("axis_lens") or [])
        nc = int(r["n_cells"])
        nt = int(r["n_trials"])
        actual = int(by_model.get(m, 0))

        print(ind2 + m)
        omit_txt = "(none — all configured axes affect this model)" if not dropped else (
            ", ".join(dropped)
        )
        omit_body = f"(fixed to grid default where skipped)  {omit_txt}"
        print(
            textwrap.fill(
                omit_body,
                width=tw,
                initial_indent=ind2 + lab_omit,
                subsequent_indent=cont,
            ),
        )

        if factors:
            lattice_txt = " × ".join(factors) + f"  =  {nc} hyperparameter cells"
            chain_txt = "×".join(str(x) for x in lens) + f"  =  {nc}"
        else:
            lattice_txt = f"(no sweep axes for this model)  =  {nc} cell(s)"
            chain_txt = "1"

        print(
            textwrap.fill(
                lattice_txt,
                width=tw,
                initial_indent=ind2 + lab_lat,
                subsequent_indent=cont,
            ),
        )
        print(
            textwrap.fill(
                chain_txt,
                width=tw,
                initial_indent=ind2 + lab_sizes,
                subsequent_indent=cont,
            ),
        )
        print(
            ind2 + f"trials = {nc} × {seeds} = {nt}     "
            f"(rows in expanded list for this model: {actual})",
        )
        print()
    print()


# ---------------------------------------------------------------------------
# Grid log — resume via jsonl
# ---------------------------------------------------------------------------

def load_completed_keys(log_path: str) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not os.path.exists(log_path):
        return done
    with open(log_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if "error" in r:
                    continue
                done.add((r["model"], r["cell_id"], int(r["seed"])))
            except Exception:
                pass
    return done


def append_log(log_path: str, record: dict) -> None:
    """Append a JSON record to the grid log, file-lock safe for parallel use."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with open(log_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------

def resolve_run_dir(explicit: str | None) -> str:
    if explicit:
        if not os.path.isdir(explicit):
            raise FileNotFoundError(f"run_dir not found: {explicit}")
        return os.path.abspath(explicit)
    datasets = scan_existing_datasets(TRAIN_DATA_DIR)
    if not datasets:
        raise RuntimeError(
            f"No preprocessed datasets found under {TRAIN_DATA_DIR}. "
            "Run `python -m Neural_Networks.apps.train` first to build one."
        )
    return datasets[0]["run_dir"]


# ---------------------------------------------------------------------------
# Config module loading
# ---------------------------------------------------------------------------

def load_grid_module(name_or_path: str):
    """Accept a bare name (laptop / hpc) or an explicit path to a .py file."""
    if os.path.isfile(name_or_path):
        path = os.path.abspath(name_or_path)
        spec = importlib.util.spec_from_file_location("_grid_cfg_adhoc", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load grid config from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.__grid_src_path__ = path
        return mod
    # Try importing as Neural_Networks.config.grids.<name>
    mod = importlib.import_module(f"Neural_Networks.config.grids.{name_or_path}")
    mod.__grid_src_path__ = getattr(mod, "__file__", "")
    return mod


def _axes_first_value_only(axes: dict) -> dict:
    """Take the first list element per axis (for auto-trial fallback)."""
    out: dict = {}
    for k, v in axes.items():
        if isinstance(v, (list, tuple)) and len(v) > 0:
            out[k] = [v[0]]
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = [v]
    return out


def _auto_trial_from_serious_module(mod) -> dict:
    """Minimal trial cfg when the preset has no ``BASE_TRIAL`` / ``AXES_TRIAL``."""
    base = dict(mod.BASE)
    base["early_stopping"] = True
    base["patience"] = int(base.get("patience", 5))
    base["epochs"] = min(int(base.get("epochs", 500)), 12)
    axes = _axes_first_value_only(dict(mod.AXES))
    n_m = min(3, len(mod.MODELS))
    models = list(mod.MODELS)[:n_m] if n_m > 0 else list(mod.MODELS)
    study = f"{getattr(mod, 'STUDY_NAME', 'grid')}_trial_auto"
    print(
        "  [grid] trial mode: preset has no BASE_TRIAL/AXES_TRIAL — "
        f"auto-trial (epochs<={base['epochs']}, first axis value each, "
        f"models={models})",
        flush=True,
    )
    return {
        "study_name":          study,
        "seeds_per_cell":      1,
        "run_dir":             getattr(mod, "RUN_DIR", None),
        "base":                base,
        "axes":                axes,
        "models":              models,
        "max_parallel_trials": getattr(mod, "MAX_PARALLEL_TRIALS", "auto"),
        "snapshot_every":      int(getattr(mod, "SNAPSHOT_EVERY", 0)),
        "_src_path":           getattr(mod, "__grid_src_path__", ""),
        "_grid_mode":          "trial",
        "_trial_source":       "auto_fallback",
    }


def grid_config_from_module(mod, *, mode: str = "serious") -> dict:
    """Build the ``cfg`` dict for ``run_grid``.

    Parameters
    ----------
    mode
        ``"serious"`` — full ``STUDY_NAME``, ``BASE``, ``AXES``, ``MODELS``.
        ``"trial"`` — ``STUDY_NAME_TRIAL``, ``BASE_TRIAL``, … when defined;
        otherwise :func:`_auto_trial_from_serious_module`.
    """
    required = ["STUDY_NAME", "SEEDS_PER_CELL", "BASE", "AXES", "MODELS"]
    for k in required:
        if not hasattr(mod, k):
            raise AttributeError(f"Grid config module missing '{k}'")
    m = mode.strip().lower()
    if m not in ("trial", "serious"):
        raise ValueError(f"mode must be 'trial' or 'serious', got {mode!r}")
    if m == "serious":
        return {
            "study_name":          mod.STUDY_NAME,
            "seeds_per_cell":      int(mod.SEEDS_PER_CELL),
            "run_dir":             getattr(mod, "RUN_DIR", None),
            "base":                dict(mod.BASE),
            "axes":                dict(mod.AXES),
            "models":              list(mod.MODELS),
            "max_parallel_trials": getattr(mod, "MAX_PARALLEL_TRIALS", "auto"),
            "snapshot_every":      int(getattr(mod, "SNAPSHOT_EVERY", 0)),
            "_src_path":           getattr(mod, "__grid_src_path__", ""),
            "_grid_mode":          "serious",
            "_trial_source":       None,
        }
    has_trial = (
        hasattr(mod, "BASE_TRIAL")
        and hasattr(mod, "AXES_TRIAL")
        and hasattr(mod, "MODELS_TRIAL")
    )
    if has_trial:
        study = getattr(mod, "STUDY_NAME_TRIAL", f"{mod.STUDY_NAME}_trial")
        snap = int(
            getattr(
                mod,
                "SNAPSHOT_TRIAL",
                getattr(mod, "SNAPSHOT_EVERY", 0),
            )
        )
        return {
            "study_name":          str(study),
            "seeds_per_cell":      int(getattr(mod, "SEEDS_TRIAL", 1)),
            "run_dir":             getattr(mod, "RUN_DIR", None),
            "base":                dict(mod.BASE_TRIAL),
            "axes":                dict(mod.AXES_TRIAL),
            "models":              list(mod.MODELS_TRIAL),
            "max_parallel_trials": getattr(mod, "MAX_PARALLEL_TRIALS", "auto"),
            "snapshot_every":      snap,
            "_src_path":           getattr(mod, "__grid_src_path__", ""),
            "_grid_mode":          "trial",
            "_trial_source":       "explicit",
        }
    return _auto_trial_from_serious_module(mod)


# ---------------------------------------------------------------------------
# Environment / metadata helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_NN_DIR, stderr=subprocess.DEVNULL, timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return None


def _gpu_info() -> dict:
    info: dict = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
    return info


def build_metadata(cfg: dict, run_dir: str, batch_dir: str,
                   trials_total: int, resumed_from: str | None) -> dict:
    return {
        "schema_version": 1,                   # bump when fields added
        "study_name":     cfg["study_name"],
        "started_at":     datetime.now().isoformat(timespec="seconds"),
        "finished_at":    None,                # filled in at the end
        "status":         "running",           # running | completed | aborted
        "resumed_from":   resumed_from,
        "batch_dir":  batch_dir,
        "dataset_run_dir": run_dir,
        "config": {
            "seeds_per_cell": cfg["seeds_per_cell"],
            "base":           cfg["base"],
            "axes":           cfg["axes"],
            "models":         cfg["models"],
            "source_path":    cfg.get("_src_path", ""),
        },
        "trials": {
            "total":   trials_total,
            "ok":      0,
            "failed":  0,
            "skipped": 0,
        },
        "env": {
            "python":   sys.version.split()[0],
            "torch":    torch.__version__,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "git_commit": _git_commit(),
            "gpu":      _gpu_info(),
        },
        "extra": {
            "grid_mode":     cfg.get("_grid_mode", "serious"),
            "trial_source":  cfg.get("_trial_source"),
        },
    }


def write_metadata(path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Seed + HP helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_hp(
    model: str,
    base: dict,
    cell_hp: dict,
    n_train_samples: int,
    snapshot_every: int = 0,
) -> dict:
    epochs = int(cell_hp.get("epochs", base.get("epochs", 500)))
    hp = get_default_hp(model, n_train_samples=n_train_samples, epochs=epochs)
    hp.update({k: v for k, v in base.items()})
    hp.update({k: v for k, v in cell_hp.items()})
    hp["snapshot_every"] = int(snapshot_every)
    return hp


# ---------------------------------------------------------------------------
# ETA estimator
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable string like ``2h 15m`` or ``43s``."""
    if seconds < 0:
        return "???"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h {m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d {h}h {m:02d}m"


class ETAEstimator:
    """Track trial completions and estimate remaining wall-clock time.

    Uses an exponential moving average (EMA) of per-trial duration so that
    ETA adapts as models with different epoch counts cycle through the queue.
    When workers > 1 the ETA accounts for parallelism.
    """

    _EMA_ALPHA = 0.3   # weight of newest sample in the moving average

    def __init__(self, total_remaining: int, n_workers: int = 1) -> None:
        self._total_remaining = total_remaining
        self._n_workers       = max(1, n_workers)
        self._completed       = 0
        self._failed          = 0
        self._ema_s: float | None = None
        self._t_start         = time.time()

    def record_completion(self, elapsed_s: float, success: bool = True) -> None:
        if success:
            self._completed += 1
        else:
            self._failed += 1
        if self._ema_s is None:
            self._ema_s = elapsed_s
        else:
            self._ema_s = self._EMA_ALPHA * elapsed_s + (1 - self._EMA_ALPHA) * self._ema_s

    @property
    def remaining(self) -> int:
        return max(0, self._total_remaining - self._completed - self._failed)

    @property
    def wall_elapsed(self) -> float:
        return time.time() - self._t_start

    @property
    def has_duration_estimate(self) -> bool:
        """True once EMA is seeded from a trial completion or epoch-1 wall snapshot."""
        return self._ema_s is not None

    def try_seed_from_epoch1_snap(self, snap: dict, n_finished_trials: int) -> None:
        """Seed ``_ema_s`` from workers' first-epoch wall time before any trial completes.

        Naive ``epoch1_wall_s × epochs_max`` assumes every epoch costs as much
        as epoch 1 (often inflated by compile/cache), which yields useless
        multi-week ETAs.  We take ``min(naive_median, median_e1×K, cap)`` where
        *K* is ``NN_GRID_ETA_SEED_K`` (default 96) and *cap* is
        ``NN_GRID_ETA_SEED_CAP_S`` (default 7200 s).  The first real
        ``record_completion`` then refines the EMA.
        """
        if n_finished_trials > 0 or self._ema_s is not None:
            return
        naive: list[float] = []
        walls: list[float] = []
        for st in snap.values():
            if not isinstance(st, dict):
                continue
            e1 = st.get("epoch1_wall_s")
            em = st.get("epochs_max")
            if not isinstance(e1, (int, float)) or float(e1) <= 0.0:
                continue
            if not isinstance(em, (int, float)) or int(em) <= 0:
                continue
            fe1 = float(e1)
            iem = int(em)
            naive.append(fe1 * float(iem))
            walls.append(fe1)
        if not naive:
            return
        median_naive = float(statistics.median(naive))
        median_e1 = float(statistics.median(walls))
        _k_raw = os.environ.get("NN_GRID_ETA_SEED_K", "96").strip()
        try:
            k_eff = max(8, int(_k_raw or "96"))
        except ValueError:
            k_eff = 96
        scaled = median_e1 * float(k_eff)
        _cap_raw = os.environ.get("NN_GRID_ETA_SEED_CAP_S", "7200").strip()
        try:
            cap_s = max(120.0, float(_cap_raw or "7200"))
        except ValueError:
            cap_s = 7200.0
        self._ema_s = min(median_naive, scaled, cap_s)
        self._ema_s = max(float(self._ema_s), 120.0)

    def eta_seconds(self) -> float:
        if self._ema_s is None or self.remaining == 0:
            return 0.0
        batches = self.remaining / self._n_workers
        return batches * self._ema_s

    def eta_seconds_from_epoch_progress(
        self, fin: int, pending_n: int, snap: dict,
    ) -> float | None:
        """Wall-clock ETA from completed trials plus fractional epoch progress.

        Unlike :meth:`eta_seconds`, this **updates every heartbeat** while
        trials run (``fin`` grows on completion; live workers contribute
        ``epoch / epochs_max``).  Returns ``None`` until enough signal exists.
        """
        elapsed = max(self.wall_elapsed, 1e-6)
        partial = 0.0
        for st in snap.values():
            if not isinstance(st, dict):
                continue
            em = int(st.get("epochs_max") or 0)
            ep = int(st.get("epoch") or 0)
            if em <= 0:
                continue
            partial += min(1.0, max(0, ep) / float(em))
        equiv_done = float(fin) + partial
        if equiv_done < 0.05:
            return None
        rate = equiv_done / elapsed
        if rate <= 1e-12:
            return None
        rem = max(0.0, float(pending_n) - equiv_done)
        return rem / rate

    def status_line(self, idx: int, total: int) -> str:
        parts = [f"[{idx:>{len(str(total))}d}/{total}]"]
        parts.append(f"Elapsed: {_fmt_duration(self.wall_elapsed)}")
        if self._ema_s is not None:
            parts.append(f"Avg: {self._ema_s:.1f}s/trial")
            parts.append(f"ETA: {_fmt_duration(self.eta_seconds())}")
        parts.append(f"Done: {self._completed}")
        if self._failed:
            parts.append(f"Failed: {self._failed}")
        parts.append(f"Left: {self.remaining}")
        if self._n_workers > 1:
            parts.append(f"Workers: {self._n_workers}")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Study-run directory resolution (timestamped by default)
# ---------------------------------------------------------------------------

def _next_batch_index(output_root: str) -> int:
    """Walk existing batch_* folders and return the next free sequential index."""
    if not os.path.isdir(output_root):
        return 1
    max_idx = 0
    for name in os.listdir(output_root):
        if not name.startswith("batch_"):
            continue
        parts = name.split("_")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        try:
            max_idx = max(max_idx, int(parts[1]))
        except ValueError:
            pass
    return max_idx + 1


def make_batch_dir(output_root: str, n_perms: int,
                   resume: str | None) -> tuple[str, str | None]:
    """Return (batch_dir, resumed_from_or_None).

    New invocation → ``batch_<NNN>_<n_perms>perms_<timestamp>/`` (NNN auto-
    increments across the whole output root so batches are globally ordered).
    Resume → caller-supplied path must already exist.
    """
    if resume:
        path = resume if os.path.isabs(resume) else os.path.join(output_root, resume)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"resume target not found: {path}")
        return os.path.abspath(path), os.path.abspath(path)

    os.makedirs(output_root, exist_ok=True)
    idx   = _next_batch_index(output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name  = f"batch_{idx:03d}_{int(n_perms)}perms_{stamp}"
    path  = os.path.join(output_root, name)
    os.makedirs(path, exist_ok=True)
    return path, None


# ---------------------------------------------------------------------------
# Subprocess worker for parallel training
# ---------------------------------------------------------------------------

def _trial_progress_key(trial: dict) -> str:
    """Stable dict key for shared parallel-grid progress (avoid ``|`` in cell_id)."""
    return "{model}\t{cell}\t{seed}".format(
        model=trial["model"],
        cell=str(trial["cell_id"]),
        seed=int(trial["seed"]),
    )


def _train_one_trial(args: dict) -> dict:
    """Top-level function executed in a spawned subprocess.

    Must be defined at module level (picklable).  Receives a plain dict with
    all the information needed to train one trial and returns a result dict.
    """
    trial           = args["trial"]
    run_dir         = args["run_dir"]
    base            = args["base"]
    n_train_samples = args["n_train_samples"]
    batch_dir       = args["batch_dir"]
    grid_registry   = args["grid_registry"]
    study_name      = args["study_name"]
    nn_dir          = args["nn_dir"]
    mem_fraction    = args["gpu_memory_fraction"]
    dl_workers      = args["dataloader_workers"]
    snapshot_every  = int(args.get("snapshot_every", 0))
    shared_progress = args.get("shared_progress")
    progress_key    = args.get("progress_key")

    # Limit dataloader subprocess count in parallel mode to avoid fork-bomb
    os.environ["NN_NUM_WORKERS"] = str(dl_workers)

    seed_everything(int(trial["seed"]))

    hp = make_hp(
        trial["model"], base, trial["cell_hp"], n_train_samples,
        snapshot_every=snapshot_every,
    )
    hp["_grid_study"]   = study_name
    hp["_grid_cell_id"] = trial["cell_id"]
    hp["_grid_seed"]    = int(trial["seed"])
    # torch.compile: only disable when GPU memory share per worker is tiny
    # (old logic used 0.9 and disabled compile for almost all parallel grids).
    try:
        _compile_cut = float(os.environ.get("NN_GRID_COMPILE_MEM_FRAC", "0.12"))
    except ValueError:
        _compile_cut = 0.12
    if mem_fraction < _compile_cut:
        hp["torch_compile"] = False

    t0 = time.time()
    try:
        save_dir, metrics = train_model(
            run_dir, trial["model"], hp,
            models_dir=batch_dir,
            registry_file=grid_registry,
            nn_dir=nn_dir,
            console=None,
            gpu_memory_fraction=mem_fraction,
            cuda_device=(
                int(args["cuda_device"])
                if args.get("cuda_device") is not None else None
            ),
            grid_progress=shared_progress,
            grid_progress_key=progress_key,
        )
        elapsed = time.time() - t0
        return {
            "success":  True,
            "model":    trial["model"],
            "cell_id":  trial["cell_id"],
            "seed":     int(trial["seed"]),
            "cell_hp":  trial["cell_hp"],
            "relevant": trial["relevant"],
            "rmse_pooled": float(metrics.get("rmse_pooled",
                                             metrics.get("rmse_mean", 0.0))),
            "rmse_mean":   float(metrics.get("rmse_mean", 0.0)),
            "r2_overall":  float(metrics.get("r2_overall", 0.0)),
            "nrmse_mean":  float(metrics.get("nrmse_mean", 0.0)),
            "epochs_trained": int(metrics.get("_epochs_trained", 0)),
            "best_epoch":     int(metrics.get("_best_epoch", 0)),
            "snapshot_epochs": list(metrics.get("_snapshot_epochs", [])),
            "save_dir":    save_dir,
            "time_s":      round(elapsed, 2),
        }
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "success":  False,
            "model":    trial["model"],
            "cell_id":  trial["cell_id"],
            "seed":     int(trial["seed"]),
            "cell_hp":  trial["cell_hp"],
            "relevant": trial["relevant"],
            "error":    str(exc),
            "traceback": traceback.format_exc(),
            "time_s":   round(elapsed, 2),
        }
    finally:
        if shared_progress is not None and progress_key:
            try:
                if progress_key in shared_progress:
                    del shared_progress[progress_key]
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_grid(
    cfg: dict,
    output_dir: str,
    models_filter: list[str] | None = None,
    dry_run: bool = False,
    resume: str | None = None,
    max_parallel_trials: int | str = "auto",
) -> str | None:
    study_name = cfg["study_name"]
    seeds      = int(cfg["seeds_per_cell"])
    base       = dict(cfg["base"])
    axes       = dict(cfg["axes"])
    models     = list(cfg["models"])
    if models_filter:
        models = [m for m in models if m in set(models_filter)]
    for m in models:
        if m not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model in grid: {m}")

    run_dir = resolve_run_dir(cfg.get("run_dir"))

    # Expand trials FIRST so the batch-dir name can embed the permutation count.
    trials = build_trial_list(models, axes, seeds)

    batch_dir, resumed_from = make_batch_dir(output_dir, len(trials), resume)

    # Snapshot the config file for reproducibility
    src = cfg.get("_src_path") or ""
    if src and os.path.isfile(src):
        try:
            shutil.copy2(src, os.path.join(batch_dir, "grid_config.py"))
        except Exception:
            pass

    log_path      = os.path.join(batch_dir, "grid_log.jsonl")
    metadata_path = os.path.join(batch_dir, "metadata.json")
    done          = load_completed_keys(log_path)

    # ---- Determine parallelism ------------------------------------------------
    plan = _parallel_worker_plan()
    if max_parallel_trials == "auto":
        base_n = int(plan["n_workers"])
    else:
        base_n = max(1, int(max_parallel_trials))
    n_workers, env_worker_bounds = _apply_env_worker_bounds(base_n)

    _n_gpus_meta = max(1, int(plan.get("n_gpus", 1)))
    _cap_pg = os.environ.get("NN_GRID_MAX_TRIALS_PER_GPU", "").strip()
    if _cap_pg.isdigit():
        _cap_n = max(1, int(_cap_pg))
        _max_w = _cap_n * _n_gpus_meta
        if n_workers > _max_w:
            print(
                f"  [grid] capping parallel workers {n_workers} → {_max_w} "
                f"(NN_GRID_MAX_TRIALS_PER_GPU={_cap_n} × {_n_gpus_meta} GPU(s))",
                flush=True,
            )
            n_workers = _max_w

    # Remaining trials after skipping already-completed ones
    pending = [t for t in trials
               if (t["model"], t["cell_id"], t["seed"]) not in done]

    meta = build_metadata(cfg, run_dir, batch_dir,
                          trials_total=len(trials), resumed_from=resumed_from)
    meta["trials"]["skipped"] = len(done)
    meta["parallel_workers"] = n_workers
    meta["extra"]["combinations"] = _combinations_summary(
        models, axes, seeds, trials,
    )
    env_par = sorted(set(plan["env_overrides"] + env_worker_bounds))
    meta["extra"]["parallelism"] = {
        "profile":             plan["profile"],
        "vram_gb":             plan["vram_gb"],
        "per_model_gb":        plan["per_model_gb"],
        "cuda_ctx_gb":         plan["cuda_ctx_gb"],
        "n_gpus":              plan.get("n_gpus", 0),
        "n_from_vram":         plan["n_from_vram"],
        "n_from_vram_per_gpu": plan.get("n_from_vram_per_gpu", plan["n_from_vram"]),
        "cpu_count":           plan["cpu_count"],
        "cpu_reserve":         plan["cpu_reserve"],
        "cpu_limit":           plan["cpu_limit"],
        "n_before_env_bounds": base_n,
        "max_parallel_trials": "auto" if max_parallel_trials == "auto" else int(
            max_parallel_trials),
        "n_workers_effective": n_workers,
        "env_overrides":       env_par,
    }
    write_metadata(metadata_path, meta)

    gpu_label = ""
    if torch.cuda.is_available():
        _ng = max(1, int(plan.get("n_gpus", 1)))
        if _ng > 1:
            gpu_label = (
                f"  [{n_workers} worker{'s' if n_workers > 1 else ''} on {_ng} GPUs "
                f"(min VRAM {plan['vram_gb']:.1f} GiB per device)]"
            )
        else:
            gpu_label = (
                f"  [{n_workers} worker{'s' if n_workers > 1 else ''} on "
                f"{torch.cuda.get_device_name(0)}]"
            )

    print(f"\nGrid study: {study_name}{gpu_label}")
    print(f"  batch dir  : {batch_dir}")
    if resumed_from:
        print(f"  (resumed)  : {resumed_from}")
    print(f"  models     : {', '.join(models)}")
    print(f"  axes       : {list(axes.keys())}")
    print(f"  seeds      : {seeds}")
    _print_combination_report(meta["extra"]["combinations"])
    print(f"  dataset    : {run_dir}")
    print(f"  workers    : {n_workers}")
    print(f"  trials     : {len(trials)}  (done: {len(done)}, "
          f"remaining: {len(pending)})\n")

    if dry_run:
        for i, t in enumerate(trials, 1):
            mark = "+" if (t["model"], t["cell_id"], t["seed"]) in done else " "
            hp_str = ", ".join(f"{k}={t['cell_hp'][k]}" for k in t["relevant"])
            print(f"  [{mark}] {i:3d}/{len(trials)}  {t['model']:30s}  seed={t['seed']}  "
                  f"cell={t['cell_id']}  [{hp_str}]")
        if not resumed_from and os.path.isdir(batch_dir) and not os.listdir(batch_dir):
            os.rmdir(batch_dir)
        return None

    # Infer n_train_samples from dataset metadata (for profile floors)
    n_train_samples = 0
    meta_json = os.path.join(run_dir, "metadata.json")
    if os.path.exists(meta_json):
        try:
            with open(meta_json) as f:
                m = json.load(f)
            n_train_samples = int((m.get("samples") or {}).get("train", 0))
        except Exception:
            pass

    grid_registry = os.path.join(batch_dir, "_grid_registry.yaml")

    # GPU memory fraction per worker (cap sums to ~usable on each GPU).
    # With multiple GPUs, trials are round-robin by pending index, so at most
    # ceil(n_workers / n_gpus) concurrent runs share one device.
    if n_workers > 1 and torch.cuda.is_available():
        n_gpus_eff = max(1, int(plan.get("n_gpus", 1)))
        vram_gb = min(
            torch.cuda.get_device_properties(i).total_memory / 1e9
            for i in range(n_gpus_eff)
        )
        ctx_gb = _cuda_ctx_gb()
        usable_frac = max(0.3, (vram_gb - ctx_gb) / vram_gb)
        n_per_gpu = max(1, (n_workers + n_gpus_eff - 1) // n_gpus_eff)
        mem_fraction = round(usable_frac / n_per_gpu, 3)
    else:
        mem_fraction = 0.95

    # Dataloader workers: small pool per trial when parallel (feed GPU faster).
    # Override with NN_GRID_DL_WORKERS (0 = main process only, like the old default).
    if n_workers > 1:
        _dl_env = os.environ.get("NN_GRID_DL_WORKERS", "").strip()
        if _dl_env.isdigit():
            dl_workers = max(0, int(_dl_env))
        else:
            _total_cpus = os.cpu_count() or 4
            dl_workers = min(
                2,
                max(0, (max(1, _total_cpus - 2)) // max(1, n_workers)),
            )
    else:
        _total_cpus = os.cpu_count() or 4
        dl_workers = max(0, min(4, _total_cpus // max(1, n_workers) - 1))

    if not pending:
        print("All trials already completed.  Nothing to do.")
        return batch_dir

    snapshot_every = int(cfg.get("snapshot_every", 0))

    # ---- Route to sequential or parallel runner -------------------------------
    if n_workers <= 1:
        ok, failed = _run_sequential(
            pending, trials, run_dir, base, n_train_samples, batch_dir,
            grid_registry, study_name, log_path, done, meta, metadata_path,
            mem_fraction, dl_workers, snapshot_every,
        )
    else:
        ok, failed = _run_parallel(
            pending, trials, run_dir, base, n_train_samples, batch_dir,
            grid_registry, study_name, log_path, done, meta, metadata_path,
            n_workers, mem_fraction, dl_workers, snapshot_every,
            max(1, int(plan.get("n_gpus", 1))),
        )

    skipped = int(len(done)) - ok
    total   = time.time() - meta.get("_t_start", time.time())

    meta["status"]      = "completed"
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["duration_s"]  = round(total, 2)
    meta["trials"].update({"ok": ok, "failed": failed, "skipped": len(done) - ok})
    write_metadata(metadata_path, meta)

    print(f"\nGrid study complete: {ok} ok, {failed} failed "
          f"(total {_fmt_duration(total)}).  Folder: {batch_dir}")

    if ok > 0:
        try:
            from Neural_Networks.apps.grid_report import run as run_report
            print("\nGenerating grid report ...")
            run_report(batch_dir, no_plots=False)
        except Exception as report_err:
            print(f"  [report warning] {report_err}")

    return batch_dir


# ---------------------------------------------------------------------------
# Sequential runner (n_workers == 1)
# ---------------------------------------------------------------------------

def _run_sequential(
    pending, trials, run_dir, base, n_train_samples, batch_dir,
    grid_registry, study_name, log_path, done, meta, metadata_path,
    mem_fraction, dl_workers, snapshot_every: int,
) -> tuple[int, int]:
    total_trials = len(trials)
    eta = ETAEstimator(total_remaining=len(pending), n_workers=1)
    ok = failed = 0

    meta["_t_start"] = time.time()

    for trial in pending:
        key = (trial["model"], trial["cell_id"], trial["seed"])
        idx = trials.index(trial) + 1

        hp_str = ", ".join(f"{k}={trial['cell_hp'][k]}" for k in trial["relevant"])
        print(f"\n[{idx:>{len(str(total_trials))}d}/{total_trials}]  "
              f"{trial['model']}  seed={trial['seed']}  cell={trial['cell_id']}")
        print(f"  {hp_str}")
        print(f"  {eta.status_line(idx, total_trials)}")
        print("-" * 100)

        seed_everything(int(trial["seed"]))

        hp = make_hp(
            trial["model"], base, trial["cell_hp"], n_train_samples,
            snapshot_every=snapshot_every,
        )
        hp["_grid_study"]   = study_name
        hp["_grid_cell_id"] = trial["cell_id"]
        hp["_grid_seed"]    = int(trial["seed"])

        os.environ["NN_NUM_WORKERS"] = str(dl_workers)

        t0 = time.time()
        try:
            save_dir, metrics = train_model(
                run_dir, trial["model"], hp,
                models_dir=batch_dir,
                registry_file=grid_registry,
                nn_dir=_NN_DIR,
                console=None,
                gpu_memory_fraction=mem_fraction,
            )
            elapsed = time.time() - t0
            record = {
                "study":       study_name,
                "model":       trial["model"],
                "cell_id":     trial["cell_id"],
                "seed":        int(trial["seed"]),
                "cell_hp":     trial["cell_hp"],
                "rmse_pooled": float(metrics.get("rmse_pooled",
                                                 metrics.get("rmse_mean", 0.0))),
                "rmse_mean":   float(metrics.get("rmse_mean", 0.0)),
                "r2_overall":  float(metrics.get("r2_overall", 0.0)),
                "nrmse_mean":  float(metrics.get("nrmse_mean", 0.0)),
                "epochs_trained": int(metrics.get("_epochs_trained", 0)),
                "best_epoch":     int(metrics.get("_best_epoch", 0)),
                "snapshot_epochs": list(metrics.get("_snapshot_epochs", [])),
                "save_dir":    save_dir,
                "time_s":      round(elapsed, 2),
                "timestamp":   datetime.now().isoformat(timespec="seconds"),
            }
            append_log(log_path, record)
            done.add(key)
            ok += 1
            eta.record_completion(elapsed, success=True)
            print(f"  >> RMSE={record['rmse_pooled']:.5f}  "
                  f"R2={record['r2_overall']:.4f}  "
                  f"best_ep={record['best_epoch']}  "
                  f"({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            traceback.print_exc()
            append_log(log_path, {
                "study":     study_name,
                "model":     trial["model"],
                "cell_id":   trial["cell_id"],
                "seed":      int(trial["seed"]),
                "cell_hp":   trial["cell_hp"],
                "error":     str(e),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            failed += 1
            eta.record_completion(elapsed, success=False)

        meta["trials"].update({"ok": ok, "failed": failed})
        meta["last_update"] = datetime.now().isoformat(timespec="seconds")
        write_metadata(metadata_path, meta)

    return ok, failed


def _status_fit(s: str, width: int) -> str:
    """Truncate or left-pad *s* to *width* characters (ASCII table cell)."""
    t = str(s).strip()
    if len(t) > width:
        return t[: max(0, width - 1)] + "."
    return t.ljust(width)


def _status_tty_clear_n_lines(n: int) -> None:
    """Move cursor up *n* lines and clear each (ANSI). No-op if *n* <= 0."""
    if n <= 0:
        return
    for _ in range(n):
        sys.stdout.write("\033[A\033[2K")
    sys.stdout.flush()


def _parallel_status_text_lines(
    ts: str,
    ok: int,
    failed: int,
    pending_n: int,
    inflight: int,
    snap: dict,
    pending_by_model: Counter,
    done_by_model: dict[str, int],
    eta: ETAEstimator,
) -> list[str]:
    """Summary line + fixed-width ASCII table for parallel grid heartbeat."""
    fin = ok + failed
    rem = max(0, pending_n - fin)
    active: dict[str, int] = {}
    min_ep: dict[str, int] = {}
    max_ep: dict[str, int] = {}
    max_mx: dict[str, int] = {}
    fracs: list[float] = []
    fracs_by_model: dict[str, list[float]] = {}
    for st in snap.values():
        if not isinstance(st, dict):
            continue
        m = str(st.get("model") or "?")
        active[m] = active.get(m, 0) + 1
        ep = int(st.get("epoch", 0) or 0)
        if m not in min_ep:
            min_ep[m] = max_ep[m] = ep
        else:
            min_ep[m] = min(min_ep[m], ep)
            max_ep[m] = max(max_ep[m], ep)
        em = st.get("epochs_max")
        if isinstance(em, (int, float)) and int(em) > 0:
            imx = int(em)
            max_mx[m] = max(max_mx.get(m, 0), imx)
            frc = ep / imx
            fracs.append(frc)
            fracs_by_model.setdefault(m, []).append(frc)
    mean_pct = 100.0 * sum(fracs) / len(fracs) if fracs else 0.0
    elapsed = _fmt_duration(eta.wall_elapsed)
    eta_prog = eta.eta_seconds_from_epoch_progress(fin, pending_n, snap)
    if rem > 0 and eta_prog is not None:
        eta_str = _fmt_duration(eta_prog)
    elif rem > 0 and eta.has_duration_estimate:
        eta_str = _fmt_duration(eta.eta_seconds())
    elif rem > 0:
        eta_str = "--"
    else:
        eta_str = "0s"
    eta_note = "ep" if eta_prog is not None else "ema"
    summary = (
        f"[grid {ts}]  {fin}/{pending_n}  ok={ok} fail={failed} rem={rem}  "
        f"up={inflight} live={len(snap)}  ep%={mean_pct:.1f}  "
        f"el {elapsed}  ETA[{eta_note}] {eta_str}"
    )
    w_m, w_d, w_t, w_a, w_e, w_p = 30, 4, 4, 3, 16, 6
    sep = (
        "+" + "-" * (w_m + 2) + "+" + "-" * (w_d + 2) + "+" + "-" * (w_t + 2)
        + "+" + "-" * (w_a + 2) + "+" + "-" * (w_e + 2) + "+" + "-" * (w_p + 2) + "+"
    )
    hdr = (
        f"| {_status_fit('Model', w_m)} | "
        f"{_status_fit('Done', w_d)} | {_status_fit('Tot', w_t)} | "
        f"{_status_fit('Act', w_a)} | {_status_fit('Epoch', w_e)} | "
        f"{_status_fit('Pct%', w_p)} |"
    )
    lines = [summary, hdr, sep]
    models = sorted(set(pending_by_model) | set(done_by_model) | set(active))
    for m in models:
        tot = int(pending_by_model.get(m, 0))
        dn = int(done_by_model.get(m, 0))
        ac = int(active.get(m, 0))
        imx = int(max_mx.get(m, 0))
        mn = min_ep.get(m, 0)
        mxep = max_ep.get(m, 0)
        if ac > 0 and imx > 0:
            if mn == mxep:
                ep_col = f"{mn}/{imx}"
            else:
                ep_col = f"{mn}-{mxep}/{imx}"
        elif ac > 0:
            ep_col = "…"
        else:
            ep_col = "-"
        fl = fracs_by_model.get(m) or []
        if fl:
            pct_str = f"{100.0 * sum(fl) / len(fl):.1f}%"
        else:
            pct_str = "-"
        if len(pct_str) > w_p:
            pct_str = pct_str[:w_p]
        row = (
            f"| {_status_fit(m, w_m)} | "
            f"{str(dn).rjust(w_d)} | {str(tot).rjust(w_t)} | {str(ac).rjust(w_a)} | "
            f"{_status_fit(ep_col, w_e)} | {pct_str.rjust(w_p)} |"
        )
        lines.append(row)
    lines.append(sep)
    _vw = os.environ.get("NN_GRID_VERBOSE_WORKERS", "").strip().lower()
    if _vw in ("1", "true", "yes", "on") and snap:
        wk = 56
        lines.append(f"| {_status_fit('worker (model / cell / seed)', wk)} | "
                     f"{_status_fit('GPU', 3)} | {_status_fit('epoch', 14)} | "
                     f"{_status_fit('val_rmse', 8)} |")
        lines.append("+" + "-" * (wk + 2) + "+" + "-" * 5 + "+" + "-" * 16
                     + "+" + "-" * 10 + "+")
        for i, (pkey, st) in enumerate(sorted(snap.items())):
            if i >= int(os.environ.get("NN_GRID_VERBOSE_WORKER_ROWS", "16")):
                lines.append(
                    f"| … {len(snap) - i} more active workers "
                    f"(raise NN_GRID_VERBOSE_WORKER_ROWS) …",
                )
                break
            if not isinstance(st, dict):
                continue
            m = str(st.get("model") or "?")
            cid = str(st.get("cell_id") or "")
            sd = int(st.get("seed") or 0)
            label = f"{m} / {cid} / {sd}"
            gpu = st.get("gpu")
            gpu_s = str(gpu) if isinstance(gpu, int) else "?"
            em = int(st.get("epochs_max") or 0)
            ep = int(st.get("epoch") or 0)
            ep_s = f"{ep}/{em}" if em > 0 else str(ep)
            vrm = st.get("val_rmse_phys")
            try:
                vrm_s = f"{float(vrm):.4f}" if vrm is not None else "-"
            except (TypeError, ValueError):
                vrm_s = "-"
            lines.append(
                f"| {_status_fit(label, wk)} | {gpu_s:>3} | {_status_fit(ep_s, 14)} | "
                f"{_status_fit(vrm_s, 8)} |",
            )
        lines.append("+" + "-" * (wk + 2) + "+" + "-" * 5 + "+" + "-" * 16
                     + "+" + "-" * 10 + "+")
    return lines


def _rich_parallel_panel(
    ts: str,
    ok: int,
    failed: int,
    pending_n: int,
    inflight: int,
    snap: dict,
    pending_by_model: Counter,
    done_by_model: dict[str, int],
    eta: ETAEstimator,
) -> Any:
    """Build a Rich renderable for ``Live`` parallel-grid status (optional)."""
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    fin = ok + failed
    rem = max(0, pending_n - fin)
    eta_prog = eta.eta_seconds_from_epoch_progress(fin, pending_n, snap)
    if rem > 0 and eta_prog is not None:
        eta_str = _fmt_duration(eta_prog)
        eta_tag = "ep"
    elif rem > 0 and eta.has_duration_estimate:
        eta_str = _fmt_duration(eta.eta_seconds())
        eta_tag = "ema"
    elif rem > 0:
        eta_str = "--"
        eta_tag = "?"
    else:
        eta_str = "0s"
        eta_tag = "done"

    summary = (
        f"{fin}/{pending_n}  ok={ok} fail={failed}  rem={rem}  "
        f"live={len(snap)}  inflight={inflight}  "
        f"wall={_fmt_duration(eta.wall_elapsed)}  ETA[{eta_tag}]={eta_str}"
    )

    tbl = Table(box=box.SIMPLE, expand=True, show_header=True, padding=(0, 1))
    tbl.add_column("Model", no_wrap=False, ratio=2)
    tbl.add_column("Done", justify="right")
    tbl.add_column("Tot", justify="right")
    tbl.add_column("Act", justify="right")
    tbl.add_column("Epoch", no_wrap=True)
    tbl.add_column("Pct", justify="right")

    active: dict[str, int] = {}
    min_ep: dict[str, int] = {}
    max_ep: dict[str, int] = {}
    max_mx: dict[str, int] = {}
    fracs_by_model: dict[str, list[float]] = {}
    for st in snap.values():
        if not isinstance(st, dict):
            continue
        m = str(st.get("model") or "?")
        active[m] = active.get(m, 0) + 1
        ep = int(st.get("epoch", 0) or 0)
        if m not in min_ep:
            min_ep[m] = max_ep[m] = ep
        else:
            min_ep[m] = min(min_ep[m], ep)
            max_ep[m] = max(max_ep[m], ep)
        em = st.get("epochs_max")
        if isinstance(em, (int, float)) and int(em) > 0:
            imx = int(em)
            max_mx[m] = max(max_mx.get(m, 0), imx)
            fracs_by_model.setdefault(m, []).append(ep / imx)

    models = sorted(set(pending_by_model) | set(done_by_model) | set(active))
    for m in models:
        tot = int(pending_by_model.get(m, 0))
        dn = int(done_by_model.get(m, 0))
        ac = int(active.get(m, 0))
        imx = int(max_mx.get(m, 0))
        mn = min_ep.get(m, 0)
        mxep = max_ep.get(m, 0)
        if ac > 0 and imx > 0:
            ep_col = f"{mn}/{imx}" if mn == mxep else f"{mn}-{mxep}/{imx}"
        elif ac > 0:
            ep_col = "…"
        else:
            ep_col = "-"
        fl = fracs_by_model.get(m) or []
        pct_str = f"{100.0 * sum(fl) / len(fl):.1f}%" if fl else "-"
        tbl.add_row(m, str(dn), str(tot), str(ac), ep_col, pct_str)

    wt = Table(box=box.SIMPLE, title="Active workers", expand=True, show_lines=False)
    wt.add_column("Trial", no_wrap=False, ratio=3)
    wt.add_column("GPU", justify="right")
    wt.add_column("Epoch", no_wrap=True)
    wt.add_column("val_rmse", justify="right")
    wt.add_column("elapsed", justify="right")
    max_rows = int(os.environ.get("NN_GRID_RICH_WORKER_ROWS", "24"))
    for i, (_pkey, st) in enumerate(sorted(snap.items())):
        if i >= max_rows:
            wt.add_row(f"… ({len(snap) - i} more)", "", "", "", "")
            break
        if not isinstance(st, dict):
            continue
        m = str(st.get("model") or "?")
        cid = str(st.get("cell_id") or "")
        sd = int(st.get("seed") or 0)
        label = f"{m} / {cid} / {sd}"
        gpu = st.get("gpu")
        gpu_s = str(gpu) if isinstance(gpu, int) else "?"
        em = int(st.get("epochs_max") or 0)
        ep = int(st.get("epoch") or 0)
        ep_s = f"{ep}/{em}" if em > 0 else str(ep)
        vrm = st.get("val_rmse_phys")
        try:
            vrm_s = f"{float(vrm):.4f}" if vrm is not None else "-"
        except (TypeError, ValueError):
            vrm_s = "-"
        tel = st.get("trial_elapsed_s")
        try:
            el_s = _fmt_duration(float(tel)) if tel is not None else "-"
        except (TypeError, ValueError):
            el_s = "-"
        wt.add_row(label, gpu_s, ep_s, vrm_s, el_s)

    parts: list[Any] = [Text.from_markup(f"[bold]{ts}[/bold]  {summary}"), tbl]
    if snap:
        parts.append(wt)
    return Panel(Group(*parts), title="Neural_Networks grid", border_style="cyan")


# ---------------------------------------------------------------------------
# Parallel runner (n_workers > 1)
# ---------------------------------------------------------------------------

def _run_parallel(
    pending, trials, run_dir, base, n_train_samples, batch_dir,
    grid_registry, study_name, log_path, done, meta, metadata_path,
    n_workers, mem_fraction, dl_workers, snapshot_every: int,
    n_gpus: int,
) -> tuple[int, int]:
    total_trials = len(trials)
    eta = ETAEstimator(total_remaining=len(pending), n_workers=n_workers)
    ok = failed = 0

    meta["_t_start"] = time.time()

    # Build worker argument dicts (all JSON-serialisable)
    worker_args_list = []
    n_gpus = max(1, int(n_gpus))
    for i, trial in enumerate(pending):
        worker_args_list.append({
            "trial":              trial,
            "run_dir":            run_dir,
            "base":               base,
            "n_train_samples":    n_train_samples,
            "batch_dir":          batch_dir,
            "grid_registry":      grid_registry,
            "study_name":         study_name,
            "nn_dir":             _NN_DIR,
            "gpu_memory_fraction": mem_fraction,
            "dataloader_workers": dl_workers,
            "snapshot_every":     snapshot_every,
            "cuda_device":        i % n_gpus,
        })

    _st_default = "2" if sys.stdout.isatty() else "15"
    _st_raw = os.environ.get("NN_GRID_STATUS_INTERVAL", _st_default).strip()
    try:
        _status_iv = max(0.5, float(_st_raw))
    except ValueError:
        _status_iv = 2.0 if sys.stdout.isatty() else 15.0

    use_tty = sys.stdout.isatty()
    use_rich = use_tty and os.environ.get(
        "NN_GRID_RICH_TUI", "",
    ).strip().lower() in ("1", "true", "yes", "on")

    print(f"Launching {n_workers} parallel workers "
          f"(mem_fraction={mem_fraction:.3f} per worker, {n_gpus} GPU(s), "
          f"dl_workers={dl_workers} per process) ...", flush=True)
    if use_rich:
        print("  status: Rich Live panel (NN_GRID_RICH_TUI)", flush=True)

    mp_ctx = multiprocessing.get_context("spawn")
    shutdown_requested = False
    manager: Any = None
    shared_progress: Any = None
    stop_hb = threading.Event()
    hb_thread: threading.Thread | None = None

    pending_by_model = Counter(t["model"] for t in pending)
    done_by_model: dict[str, int] = {}
    status_lock = threading.Lock()
    hb_line_count = {"n": 0}

    try:
        manager = multiprocessing.Manager()
        shared_progress = manager.dict()
        for wa in worker_args_list:
            wa["shared_progress"] = shared_progress
            wa["progress_key"] = _trial_progress_key(wa["trial"])

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_ctx,
        ) as executor:
            future_to_trial: dict = {}
            for wa in worker_args_list:
                fut = executor.submit(_train_one_trial, wa)
                future_to_trial[fut] = wa["trial"]

            def _heartbeat_ascii() -> None:
                first = True
                _first_wait = min(2.0, _status_iv)
                while not stop_hb.wait(_first_wait if first else _status_iv):
                    first = False
                    try:
                        snap = dict(shared_progress)
                    except Exception:
                        snap = {}
                    inflight = sum(
                        1 for f in future_to_trial
                        if not f.done() and not f.cancelled()
                    )
                    ts = datetime.now().strftime("%H:%M:%S")
                    eta.try_seed_from_epoch1_snap(snap, ok + failed)
                    lines = _parallel_status_text_lines(
                        ts, ok, failed, len(pending), inflight, snap,
                        pending_by_model, done_by_model, eta,
                    )
                    with status_lock:
                        if use_tty and not use_rich and hb_line_count["n"] > 0:
                            _status_tty_clear_n_lines(hb_line_count["n"])
                        if use_tty:
                            for ln in lines:
                                print(ln, flush=True)
                            if not use_rich:
                                hb_line_count["n"] = len(lines)
                        else:
                            for ln in lines:
                                print(ln, flush=True)

            def _heartbeat_rich() -> None:
                try:
                    from rich.console import Console
                    from rich.live import Live
                except Exception:
                    _heartbeat_ascii()
                    return
                first = True
                _first_wait = min(2.0, _status_iv)
                try:
                    console = Console(file=sys.stdout, force_terminal=True)
                except Exception:
                    _heartbeat_ascii()
                    return
                with Live(
                    console=console,
                    refresh_per_second=min(8.0, 1.0 / max(_status_iv * 0.5, 0.05)),
                    transient=True,
                ) as live:
                    while not stop_hb.wait(_first_wait if first else _status_iv):
                        first = False
                        try:
                            snap = dict(shared_progress)
                        except Exception:
                            snap = {}
                        inflight = sum(
                            1 for f in future_to_trial
                            if not f.done() and not f.cancelled()
                        )
                        ts = datetime.now().strftime("%H:%M:%S")
                        eta.try_seed_from_epoch1_snap(snap, ok + failed)
                        panel = _rich_parallel_panel(
                            ts, ok, failed, len(pending), inflight, snap,
                            pending_by_model, done_by_model, eta,
                        )
                        with status_lock:
                            live.update(panel)

            hb_thread = threading.Thread(
                target=_heartbeat_rich if use_rich else _heartbeat_ascii,
                daemon=True,
            )
            hb_thread.start()

            try:
                for fut in concurrent.futures.as_completed(future_to_trial):
                    trial = future_to_trial[fut]
                    key   = (trial["model"], trial["cell_id"], trial["seed"])
                    idx   = trials.index(trial) + 1

                    try:
                        result = fut.result()
                    except Exception as exc:
                        result = {
                            "success":  False,
                            "model":    trial["model"],
                            "cell_id":  trial["cell_id"],
                            "seed":     int(trial["seed"]),
                            "cell_hp":  trial["cell_hp"],
                            "relevant": trial["relevant"],
                            "error":    str(exc),
                            "time_s":   0.0,
                        }

                    elapsed = float(result.get("time_s", 0.0))

                    _mdl = str(result.get("model") or "")
                    if _mdl:
                        done_by_model[_mdl] = done_by_model.get(_mdl, 0) + 1

                    if result["success"]:
                        record = {
                            "study":          study_name,
                            "model":          result["model"],
                            "cell_id":        result["cell_id"],
                            "seed":           result["seed"],
                            "cell_hp":        result["cell_hp"],
                            "rmse_pooled":    result["rmse_pooled"],
                            "rmse_mean":      result["rmse_mean"],
                            "r2_overall":     result["r2_overall"],
                            "nrmse_mean":     result["nrmse_mean"],
                            "epochs_trained": result["epochs_trained"],
                            "best_epoch":     result["best_epoch"],
                            "snapshot_epochs": list(result.get("snapshot_epochs", [])),
                            "save_dir":       result.get("save_dir", ""),
                            "time_s":         result["time_s"],
                            "timestamp":      datetime.now().isoformat(timespec="seconds"),
                        }
                        append_log(log_path, record)
                        done.add(key)
                        ok += 1
                        eta.record_completion(elapsed, success=True)

                        hp_str = ", ".join(
                            f"{k}={result['cell_hp'].get(k, '?')}"
                            for k in result.get("relevant", [])
                        )
                        with status_lock:
                            if use_tty and not use_rich and hb_line_count["n"] > 0:
                                _status_tty_clear_n_lines(hb_line_count["n"])
                                hb_line_count["n"] = 0
                            print(f"[{idx:>{len(str(total_trials))}d}/{total_trials}]  "
                                  f"OK  {result['model']}  seed={result['seed']}  "
                                  f"RMSE={result['rmse_pooled']:.5f}  "
                                  f"R2={result['r2_overall']:.4f}  "
                                  f"best_ep={result['best_epoch']}  "
                                  f"({elapsed:.1f}s)")
                            print(f"  {hp_str}")
                            print(f"  {eta.status_line(ok + failed, len(pending))}")
                    else:
                        append_log(log_path, {
                            "study":     study_name,
                            "model":     result["model"],
                            "cell_id":   result["cell_id"],
                            "seed":      result["seed"],
                            "cell_hp":   result["cell_hp"],
                            "error":     result.get("error", "unknown"),
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        })
                        failed += 1
                        eta.record_completion(elapsed, success=False)
                        with status_lock:
                            if use_tty and not use_rich and hb_line_count["n"] > 0:
                                _status_tty_clear_n_lines(hb_line_count["n"])
                                hb_line_count["n"] = 0
                            print(f"[{idx:>{len(str(total_trials))}d}/{total_trials}]  "
                                  f"FAIL  {result['model']}  seed={result['seed']}  "
                                  f"({elapsed:.1f}s)  {result.get('error', '')[:120]}")
                            if result.get("traceback"):
                                for line in result["traceback"].strip().splitlines()[-5:]:
                                    print(f"    {line}")
                            print(f"  {eta.status_line(ok + failed, len(pending))}")

                    # Refresh metadata after every completion
                    meta["trials"].update({"ok": ok, "failed": failed})
                    meta["last_update"] = datetime.now().isoformat(timespec="seconds")
                    write_metadata(metadata_path, meta)

            except KeyboardInterrupt:
                shutdown_requested = True
                with status_lock:
                    if use_tty and not use_rich and hb_line_count["n"] > 0:
                        _status_tty_clear_n_lines(hb_line_count["n"])
                        hb_line_count["n"] = 0
                    print("\n\nKeyboardInterrupt received -- cancelling pending trials ...",
                          flush=True)
                for fut in future_to_trial:
                    fut.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
            finally:
                stop_hb.set()
                if hb_thread is not None:
                    hb_thread.join(timeout=5.0)
                with status_lock:
                    if use_tty and not use_rich and hb_line_count["n"] > 0:
                        _status_tty_clear_n_lines(hb_line_count["n"])
                        hb_line_count["n"] = 0
                    sys.stdout.write("\n")
                    sys.stdout.flush()

    finally:
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass

    if shutdown_requested:
        meta["status"] = "aborted"
        meta["last_update"] = datetime.now().isoformat(timespec="seconds")
        write_metadata(metadata_path, meta)
        print(f"Aborted after {ok} ok, {failed} failed.  "
              f"Resume with the same batch dir to continue.")

    return ok, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_grid_mode(cli_mode: str | None) -> str:
    """``--mode`` > ``NN_GRID_MODE`` > TTY prompt > serious (non-TTY default)."""
    if cli_mode in ("trial", "serious"):
        return cli_mode
    env_m = os.environ.get("NN_GRID_MODE", "").strip().lower()
    if env_m in ("trial", "serious"):
        return env_m
    if sys.stdin.isatty() and sys.stdout.isatty():
        print("\nGrid mode:", flush=True)
        print("  [1] Trial  — quick smoke (small sweep)", flush=True)
        print("  [2] Serious — full Cartesian sweep", flush=True)
        for _ in range(4):
            try:
                raw = input("Choice (1 or 2, default 2): ").strip().lower()
            except EOFError:
                return "serious"
            if raw in ("", "2", "s", "serious"):
                return "serious"
            if raw in ("1", "t", "trial"):
                return "trial"
            print("  Please enter 1 or 2.", flush=True)
        return "serious"
    return "serious"


def main(argv: list[str] | None = None) -> int:
    """Load grid preset and run.  See module docstring for preset selection."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter grid search (calls core.trainer only).",
    )
    parser.add_argument(
        "--mode",
        choices=("trial", "serious"),
        default=None,
        help="trial = small sweep; serious = full preset (overrides NN_GRID_MODE)",
    )
    args = parser.parse_args(argv)

    active_mod = importlib.import_module("Neural_Networks.config.grids.active")

    preset_env = os.environ.get("NN_GRID_PRESET", "").strip().lower()
    if preset_env in ("hpc", "laptop"):
        mod = importlib.import_module(
            f"Neural_Networks.config.grids.{preset_env}",
        )
        mod.__grid_src_path__ = getattr(mod, "__file__", "")
        print(f"preset: {preset_env}  (NN_GRID_PRESET override)")
    elif _auto_grid_preset_name() == "hpc":
        mod = importlib.import_module("Neural_Networks.config.grids.hpc")
        mod.__grid_src_path__ = getattr(mod, "__file__", "")
        try:
            _prof = detect_hardware().get("profile", "")
        except Exception:
            _prof = ""
        if _prof == "server":
            print("preset: hpc  (auto: server-class GPU, VRAM >= 40 GB)")
        elif _has_batch_scheduler():
            print("preset: hpc  (auto: batch scheduler + VRAM >= 40 GB)")
        else:
            print("preset: hpc  (auto)")
    else:
        mod = active_mod
        mod.__grid_src_path__ = getattr(mod, "__file__", "")
        print("preset: active.py  (local / non-server; set NN_GRID_PRESET to force)")

    grid_mode = _resolve_grid_mode(args.mode)
    print(f"grid_mode: {grid_mode}", flush=True)
    cfg = grid_config_from_module(mod, mode=grid_mode)

    dry_run       = bool(getattr(active_mod, "DRY_RUN", False))
    models_filter = getattr(active_mod, "MODELS_FILTER", None)
    resume        = getattr(active_mod, "RESUME", None)

    max_par = getattr(
        active_mod, "MAX_PARALLEL_TRIALS",
        cfg.get("max_parallel_trials", "auto"),
    )

    run_grid(cfg, GRID_OUTPUT_DIR,
             models_filter=models_filter,
             dry_run=dry_run,
             resume=resume,
             max_parallel_trials=max_par)
    return 0


if __name__ == "__main__":
    sys.exit(main())
