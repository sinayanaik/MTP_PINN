#!/usr/bin/env python3
"""Local hyperparameter grid search for the three torque model variants.

Models compared:
    * ``BlackBoxFNN``            — purely data-driven MLP baseline.
    * ``PhysicsRegularizedFNN``  — MLP + learnable calibration on summed
                                    analytical torque; blended data / physics loss.
    * ``ResidualCorrectionFNN``  — predicts Δ on top of τ_phys; magnitude-penalty.

Research goal: does physics inductive bias beat the BlackBox baseline, and does
the advantage grow as training data shrinks?  The grid is deliberately compact
(144 trials = 12 FNN + 72 PhysReg + 60 Residual) covering six data fracs
(full → 2%) × six physics-weight / five alpha values × two seeds for mean±spread.

Execution strategy — fully dynamic, resource-polling parallelism
----------------------------------------------------------------
The runner picks the number of concurrent trials *at every admission decision*
by reading live free VRAM and RAM.  No fixed concurrency count, no fixed
cooldown between trials.  The admission loop:

    1. Estimates each pending trial's VRAM + RAM from its own hyperparameters.
    2. Polls the actual free VRAM / RAM.
    3. Launches the next trial only if it fits (with a reserve for OS/display).
    4. Otherwise sleeps briefly and re-checks — naturally handling the slow
       CUDA-context release that has crashed this machine in the past.

On CPU-only systems the runner is sequential (parallel CPU trials wreck each
other via OMP thread contention).  On GPU systems with enough VRAM for only
one trial, the runner is sequential (no pool overhead).

Edit config constants below and run::

    PYTHONPATH=. python3 -m Neural_Networks.models.run_loss_residual_grid

Results land in ``Trained_Models_Grid/`` (separate from ``Trained_Models/``).
Analyse afterwards with::

    PYTHONPATH=. python3 -m Neural_Networks.analyze_models_grid
"""

from __future__ import annotations

import gc
import itertools
import logging
import math
import multiprocessing as mp
import os
import sys
import threading
import time
import warnings
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# torch is imported lazily — workers set CUDA_VISIBLE_DEVICES before import.
from tqdm import tqdm

# ============================================================================
# ── CONFIGURATION  (edit here — no CLI arguments) ────────────────────────────
# ============================================================================

# Which architectures to sweep.
#   "all"      → BlackBoxFNN + PhysicsRegularizedFNN + ResidualCorrectionFNN
#   "fnn" | "physreg" | "residual" → just that one
ARCH: str = "all"

# Print the combo table and exit without training.
DRY_RUN: bool = False

# Skip combos whose output dir already contains a matching metadata.yaml.
SKIP_EXISTING: bool = True

# Dataset run directory (pre-processed CSV produced by preprocess_data.py GUI).
_NN_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_RUN_DIR: str = os.environ.get("MTP_TRAIN_DATA_RUN") or str(
    _NN_ROOT / "train_data"
    / "run_0425_1112_qraw_d25p3i_ddL_mraw_a1_R_70v15t15_f1p0t1p0_789d82"
)

# Output root — completely separate from Trained_Models/.
# Runs land under  Trained_Models_Grid/<dataset_name>/ModelType/<run>/
# so every dataset's results stay grouped together and analysis can target
# a single dataset folder via --models-dir.
MODELS_DIR_ROOT: str = str(_NN_ROOT / "Trained_Models_Grid")

# Dataset-scoped output root — derived from TRAIN_DATA_RUN_DIR at runtime.
_DATASET_NAME:  str = Path(TRAIN_DATA_RUN_DIR).name
DATASET_OUT_ROOT: str = str(Path(MODELS_DIR_ROOT) / _DATASET_NAME)
REGISTRY_FILE:   str = str(Path(DATASET_OUT_ROOT) / "models_registry.yaml")

# Batch size for every trial. The memory estimator scales accordingly — drop
# to 512 if the admission loop reports "tight resources" repeatedly.
BATCH_SIZE: int = 1024

# ── Resource-admission parameters (no hardcoded concurrency) ─────────────────
# Reserves: NEVER consumed by the grid (keeps the desktop responsive).
# RAM reserve was 2.0 GB; on 16 GB laptops with a heavy IDE/browser already
# running, that combined with a 1.0 GB per-trial RAM estimate meant the
# admission loop needed 3+ GB free just to dispatch one trial. Workers would
# finish their first task and then stall indefinitely because free RAM
# (post-worker-RSS) never climbed back to 3 GB. 1.0 GB is adequate protection.
VRAM_RESERVE_GB: float = 0.5
RAM_RESERVE_GB:  float = 1.0

# Polling cadence for the admission loop.
ADMISSION_POLL_SEC: float = 1.0   # while in-flight trials are running
TIGHT_SLEEP_SEC:    float = 5.0   # when nothing currently fits

# Sequential-path RAM floor (laptop-grade check).
MIN_FREE_RAM_GB:   float = 2.0
MEM_POLL_INTERVAL: float = 5.0

# ============================================================================
# ── FIXED HYPER-PARAMETERS (identical for every trial) ───────────────────────
# ============================================================================

FIXED_HP: dict[str, Any] = {
    # Architecture (constant — we're comparing *variants*, not sizes).
    # Smaller model (128-256-128 ≈ 95K params vs 267K) reduces overfitting
    # with typical training set sizes of 8K–50K samples.
    "hidden_layers":           [128, 256, 128],
    "dropout":                 0.2,
    "activation":              "gelu",

    # Optimisation.
    "learning_rate":           3e-4,
    "weight_decay":            5e-2,
    "batch_size":              BATCH_SIZE,
    "lr_scheduler":            "warmup_cosine",
    "grad_clip_norm":          1.0,
    "feature_noise_std":       0.05,

    # Training length / early stopping.
    "epochs":                  3000,
    "patience":                150,
    "min_delta":               1e-4,
    "early_stopping":          True,
    "early_stop_metric":       "val_rmse",

    # Use .npy sidecars via memmap when they exist (see tools/convert_run_csv_to_npy).
    # Falls back to CSV loading automatically if the sidecars are missing.
    "dataset_memmap":          True,

    # Bookkeeping.
    "stride":                  1,

    # Seed defaults (overridden per-trial by the sweep).
    "seed":                    0,
    "data_train_seed":         0,

    # PhysReg-only HPs (ignored by other strategies).
    "physics_warmup_fraction": 0.05,

    # ResidualCorrectionFNN: fixed tanh bound on correction magnitude.
    # Limits corrections to ±correction_scale in normalised torque units.
    "correction_scale":        0.5,
}

# ============================================================================
# ── HYPERPARAMETER SWEEPS ────────────────────────────────────────────────────
#
# Goal: isolate (data availability) × (physics weighting) for each architecture.
# Everything else is fixed; seed gives two replicates per combo.
# ============================================================================

# Minimal grid focused on a single question: does PINN inductive bias beat the
# BlackBox, and how does the answer change as training data gets scarcer?
# Three data regimes (full, quarter, tenth), three physics-strength regimes
# per PINN variant (off, moderate, strong), two seeds for mean±spread.
_SEEDS      = [0, 1]
_DATA_FRACS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]  # full → 2% (exposes data-efficiency regime)

# FNN baseline: data × seed.  6 × 2 = 12 combos.
GRID_FNN: dict[str, list] = {
    "data_train_fraction": _DATA_FRACS,
    "seed":                _SEEDS,
}

# Physics-regularized: physics_weight × data × seed.  6 × 6 × 2 = 72 combos.
# Loss = L_data + pw * L_phys  (additive Tikhonov — pw is the penalty coefficient).
# pw=0.05 → weak physics nudge (near-BlackBox sanity check).
# pw=0.5  → moderate; expect RMSE improvement vs BlackBox for all data fractions.
# pw=2.0  → strong prior; network predictions tightly constrained to physics.
GRID_PHYSREG: dict[str, list] = {
    "physics_weight":      [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    "data_train_fraction": _DATA_FRACS,
    "seed":                _SEEDS,
}

# Physics-residual: alpha_reg × data × seed.  5 × 6 × 2 = 60 combos.
# tanh bounding (correction_scale=0.5) provides hard structural constraint;
# alpha_reg is the additional L2 penalty on the bounded correction magnitude.
# alpha=0.005 → minimal additional penalty (structural bound dominates).
# alpha=0.1   → moderate L2 tightening.
# alpha=0.5   → strong L2; expect near-physics predictions.
GRID_RESIDUAL: dict[str, list] = {
    "alpha_reg_weight":    [0.005, 0.01, 0.05, 0.1, 0.5],
    "data_train_fraction": _DATA_FRACS,
    "seed":                _SEEDS,
}

# Total: 12 + 72 + 60 = 144 trials when ARCH="all".

_ARCH_META: dict[str, tuple[str, str, str]] = {
    "fnn":      ("BlackBoxFNN",           "FNN",                   "Neural_Networks/models/run_fnn.py"),
    "physreg":  ("PhysicsRegularizedFNN", "PhysicsRegularizedFNN", "Neural_Networks/models/run_physics_regularized.py"),
    "residual": ("ResidualCorrectionFNN", "ResidualCorrectionFNN", "Neural_Networks/models/run_physics_residual.py"),
}

_ARCH_GRID: dict[str, dict[str, list]] = {
    "fnn":      GRID_FNN,
    "physreg":  GRID_PHYSREG,
    "residual": GRID_RESIDUAL,
}

# HP keys excluded from the "already-trained?" fingerprint (hardware-dependent
# or historical).  Keys starting with ``_`` are also excluded.
_SKIP_KEYS = frozenset({
    "torch_compile", "torch_compile_mode", "_grid_seed",
    "phi_lr_ratio", "optimizer", "snapshot_every", "print_every",
})


# ============================================================================
# ── TRIAL-BUILDING HELPERS ───────────────────────────────────────────────────
# ============================================================================

def _cartesian(grid: dict[str, list]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def _build_trials() -> list[dict[str, Any]]:
    active_archs = list(_ARCH_META.keys()) if ARCH == "all" else [ARCH]
    if not all(a in _ARCH_META for a in active_archs):
        raise ValueError(f"ARCH={ARCH!r} must be 'all' or one of {list(_ARCH_META)}")

    trials: list[dict[str, Any]] = []
    for arch in active_archs:
        model_type, save_subdir, run_help = _ARCH_META[arch]
        for combo in _cartesian(_ARCH_GRID[arch]):
            hp = {**FIXED_HP, **combo}
            # Seed axis populates both torch init seed and data subsample seed
            # so each replicate is a *complete* re-run.
            if "seed" in combo:
                hp["data_train_seed"] = int(combo["seed"])
            # Version marker: physreg/residual have a new in_dim (15→20) — old
            # checkpoints are incompatible.  This key is NOT in _SKIP_KEYS so
            # existing metadata without it won't fingerprint-match → re-trained.
            if arch in ("physreg", "residual"):
                hp["phys_input_concat"] = "7J"  # 7J input; old "True" runs (4J) must re-run
            trials.append({
                "arch":        arch,
                "model_type":  model_type,
                "save_subdir": save_subdir,
                "run_help":    run_help,
                "hp":          hp,
            })
    return trials


def _partition_trials_by_skip(trials: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Pre-filter trials against DATASET_OUT_ROOT. Returns (pending, skip_count).

    Runs in the main process before any worker is spawned. Saves the 3–5 s
    cold-start-per-skipped-trial cost that was paid when the check lived
    inside the worker.
    """
    if not SKIP_EXISTING:
        return list(trials), 0
    pending: list[dict[str, Any]] = []
    skipped = 0
    subdir_cache: dict[str, Path] = {}
    for t in trials:
        sd = t["save_subdir"]
        base = subdir_cache.get(sd)
        if base is None:
            base = Path(DATASET_OUT_ROOT) / sd
            subdir_cache[sd] = base
        if base.is_dir() and _find_existing_run(base, t["model_type"], t["hp"]):
            skipped += 1
        else:
            pending.append(t)
    return pending, skipped


def _find_existing_run(subdir_path: Path, model_type: str, hp: dict) -> bool:
    """Return True if *subdir_path* already holds a run matching this HP set."""
    import yaml
    compare_hp = {
        k: v for k, v in hp.items()
        if k not in _SKIP_KEYS and not k.startswith("_")
    }
    for meta_file in subdir_path.glob("*/metadata.yaml"):
        try:
            with open(meta_file) as fh:
                m = yaml.safe_load(fh)
            if not isinstance(m, dict) or m.get("model_type") != model_type:
                continue
            saved = {
                k: v for k, v in m.get("hyperparams", {}).items()
                if k not in _SKIP_KEYS and not k.startswith("_")
            }
            if saved == compare_hp:
                return True
        except Exception:
            pass
    return False


def _hp_desc(arch: str, hp: dict) -> str:
    extras = {"physics_weight": "pw", "alpha_reg_weight": "arw"}
    extra = "  ".join(f"{short}={hp[k]}" for k, short in extras.items() if k in hp)
    base = (
        f"{arch:<8} frac={hp.get('data_train_fraction','?')} "
        f"seed={hp.get('seed','?')}"
    )
    return f"{base}  {extra}" if extra else base


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _print_combo_table(trials: list[dict[str, Any]]) -> None:
    counts = Counter(t["arch"] for t in trials)
    print(f"\n{'='*76}")
    print(f"  GRID SEARCH  —  ARCH={ARCH!r}   Total={len(trials)} combos")
    for arch, cnt in sorted(counts.items()):
        print(f"    {arch:<12}  {cnt} combos")
    print(f"{'='*76}")
    print(f"  {'#':<4} {'arch':<10} {'frac':<6} {'seed':<5} extras")
    print(f"  {'-'*72}")
    for i, t in enumerate(trials, 1):
        hp = t["hp"]
        extra = "  ".join(
            f"{k}={hp[k]}" for k in ("physics_weight", "alpha_reg_weight") if k in hp
        )
        print(
            f"  {i:<4} {t['arch']:<10} {str(hp.get('data_train_fraction','?')):<6} "
            f"{str(hp.get('seed','?')):<5} {extra}"
        )
    print(f"{'='*76}\n")


# ============================================================================
# ── RESOURCE QUERIES (live, dynamic) ─────────────────────────────────────────
# ============================================================================

# Input dims: FNN uses kinematics only (3J=15), physics models use 7J=35.
_N_IN      = 15   # BlackBoxFNN: [q, qd, qdd]
_N_IN_PHYS = 35   # PhysicsRegularizedFNN / ResidualCorrectionFNN: [q, qd, qdd, τ_g, τ_M, τ_C, τ_f]
_N_OUT     = 5


def _count_params(hidden_layers: list[int], n_in: int = _N_IN, extra: int = 0) -> int:
    """Match :func:`torque_models.build_mlp` exactly (Linear + LayerNorm)."""
    dims = [n_in] + list(hidden_layers) + [_N_OUT]
    total = 0
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        total += a * b + b          # Linear(a→b): weight + bias
        if i < len(dims) - 2:
            total += 2 * b          # LayerNorm(b): weight + bias
    return total + extra


# Framework overhead on top of the analytical parameter-count formula:
# cuDNN workspaces, cuBLAS buffers, PyTorch allocator's steady-state caching.
# Measured empirically for the 256-512-256 MLP at bs=1024: ~80-120 MB above the
# analytical estimate.  0.5 GB (previous default) was ~4x too conservative on a
# 4 GB card and held admission to 3 workers even with plenty of headroom.
_FRAMEWORK_VRAM_OVERHEAD_GB: float = 0.15


def _estimate_trial_mem(hp: dict, arch: str, cuda_ctx_gb: float) -> tuple[float, float]:
    """Conservative (vram_gb, ram_gb) estimate for one trial's own HP."""
    hl = hp.get("hidden_layers", [128, 256, 128])
    bs = int(hp.get("batch_size", 1024))
    n_in = _N_IN_PHYS if arch in ("physreg", "residual") else _N_IN
    P = _count_params(hl, n_in=n_in, extra=0)

    B = 4                                                    # float32 bytes
    model_b = P * B
    grad_b  = P * B
    adam_b  = P * 2 * B                                      # exp_avg + exp_avg_sq
    act_b   = sum(hl) * bs * B * 2                           # fwd + bwd activations
    io_b    = bs * (n_in + _N_OUT) * B

    raw = (model_b + grad_b + adam_b + act_b + io_b) * 1.5   # allocator headroom
    vram_gb = raw / 1e9 + cuda_ctx_gb + _FRAMEWORK_VRAM_OVERHEAD_GB
    # Marginal RAM cost of one trial on top of a LIVE worker (Python + torch
    # + memmapped dataset already resident).  Previously this was +1.0 GB
    # representing FRESH-worker RSS — but after initial fill all admissions
    # target reused workers, and charging the full 1 GB again made admission
    # stall as soon as free RAM dipped below ~3 GB.  0.3 GB covers the tensor
    # working set for one trial.
    ram_gb  = raw / 1e9 + 0.3
    return vram_gb, ram_gb


def _query_cuda_available() -> bool:
    """True if CUDA is usable.  Cheap — does not create a CUDA context."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _query_n_gpus() -> int:
    """Count CUDA-visible GPUs reported by nvidia-smi."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        return max(1, len([ln for ln in out.splitlines() if ln.strip()]))
    except Exception:
        return 1


def _query_free_vram_gb_per_gpu() -> list[float]:
    """Free VRAM per GPU in GB, indexed by GPU ordinal."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        return [float(ln.strip()) / 1024.0 for ln in out.splitlines() if ln.strip()]
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                return [torch.cuda.mem_get_info(i)[0] / 1e9
                        for i in range(torch.cuda.device_count())]
        except Exception:
            pass
        return [0.0]


def _query_free_vram_gb() -> float:
    """Total free VRAM across all GPUs in GB."""
    return sum(_query_free_vram_gb_per_gpu())


def _query_total_vram_gb() -> float:
    """Total VRAM across all GPUs in GB."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return sum(float(ln) / 1024.0 for ln in lines)
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                return sum(
                    torch.cuda.get_device_properties(i).total_memory / 1e9
                    for i in range(torch.cuda.device_count())
                )
        except Exception:
            pass
        return 0.0


def _query_free_ram_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 1e9


def _measure_cuda_ctx_gb() -> float:
    """Measure a fresh process's CUDA context overhead via a child subprocess.

    We do NOT touch CUDA in the main process, because even a single context
    permanently consumes ~350 MB on this GPU — non-trivial on a 4 GB card.
    The child allocates a 1-element tensor, prints the VRAM delta, and exits.
    """
    import subprocess
    import sys
    script = (
        "import torch, sys\n"
        "if not torch.cuda.is_available():\n"
        "    print(0); sys.exit(0)\n"
        "f0, _ = torch.cuda.mem_get_info(0)\n"
        "t = torch.zeros(1, device='cuda:0')\n"
        "f1, _ = torch.cuda.mem_get_info(0)\n"
        "del t\n"
        "print((f0 - f1) / 1e9)\n"
    )
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", script], timeout=30, text=True,
        ).strip()
        val = float(out.splitlines()[-1])
        # Defensive: if the measurement looks absurd, fall back to a typical value.
        return val if 0.05 <= val <= 2.0 else 0.35
    except Exception:
        return 0.35


def _compute_pool_size(cuda_available: bool) -> int:
    """Derive an upper bound on concurrent workers from hardware alone.

    This is the *pool capacity*, not a concurrency target; the admission loop
    decides how many are actually in flight based on live free VRAM / RAM.
    """
    import psutil
    if not cuda_available:
        return 1                                         # CPU ⇒ sequential
    cpu_phys = psutil.cpu_count(logical=False) or 2
    vram_total = _query_total_vram_gb()
    # One slot per ~0.8 GB of total VRAM (CUDA ctx ~0.35 + small MLP working
    # set ~0.3 GB + framework overhead ~0.15).  4 GB → 5 slots, 8 GB → 10,
    # 16 GB → 20.  Live-admission loop still enforces per-trial fit, so this
    # is a concurrency ceiling only.
    n_vram = max(1, int(vram_total / 0.8))               # 5 on 4 GB, 10 on 8 GB
    # CPU budget: leave 2 physical cores for OS + main process + drain thread.
    # threads_per_worker has a floor of 2 (single-thread torch is flaky).
    # cpu_phys - 2 gives: 4-core → 2, 6-core → 4, 8-core → 6, matching the
    # actual parallelism the machine can sustain with small MLP training.
    n_cpu  = max(2, cpu_phys - 2)
    # Optional override: ``MTP_GRID_POOL_SIZE=N`` forces N regardless of heuristics.
    # Useful when you know your workload is light enough to oversubscribe.
    override = os.environ.get("MTP_GRID_POOL_SIZE", "").strip()
    if override.isdigit() and int(override) >= 1:
        return int(override)
    return min(n_vram, n_cpu)


def _compute_threads_per_worker(pool_size: int) -> int:
    import psutil
    cpu_phys = psutil.cpu_count(logical=False) or 2
    # Leave 1 physical core free for the OS / display.  Minimum 2 threads —
    # single-threaded torch ops have been linked to intermittent deadlocks.
    return max(2, (cpu_phys - 1) // max(1, pool_size))


# ============================================================================
# ── ENVIRONMENT DETECTION ────────────────────────────────────────────────────
# Classify the machine so callers can pick compile flags and pool parameters
# without hardcoding hardware assumptions.
# ============================================================================

def _detect_env() -> str:
    """Return 'hpc', 'workstation', or 'laptop' based on measured hardware.

    Thresholds (all three must pass for the higher tier):
      hpc         : cpu_phys >= 16  AND  ram_gb >= 32  AND  vram_gb >= 16
      workstation : cpu_phys >= 8   AND  ram_gb >= 16  AND  vram_gb >= 6
      laptop      : everything else
    """
    try:
        import psutil
        cpu_phys = psutil.cpu_count(logical=False) or 1
        ram_gb   = psutil.virtual_memory().total / 1e9
        vram_gb  = _query_total_vram_gb()
        if cpu_phys >= 16 and ram_gb >= 32 and vram_gb >= 16:
            return "hpc"
        if cpu_phys >= 8  and ram_gb >= 16 and vram_gb >= 6:
            return "workstation"
    except Exception:
        pass
    return "laptop"


def _compile_flags(env: str, arch: str) -> tuple[bool, str]:
    """Return (torch_compile_enabled, mode) appropriate for env + arch.

    torch.compile with reduce-overhead uses CUDA graphs: after a short warm-up
    (~30 s on first trial) it replays the captured graph with near-zero Python
    overhead — typically 20–40% faster for small fixed-shape MLPs on Ampere+ GPUs.
    Not beneficial on laptops where the compile cost dwarfs the savings.

    EDR uses autograd Jacobians which are incompatible with graph capture.
    """
    if arch == "edr":
        return False, "default"            # Jacobian capture breaks torch.compile
    if env == "hpc":
        return True, "reduce-overhead"     # CUDA graphs, best for A100/H100
    if env == "workstation":
        return True, "default"             # safe general mode, mild speedup
    return False, "default"                # laptop: compile overhead > savings


def _maxtasks_for_env(env: str) -> int:
    """Worker reuse limit before the process is recycled.

    Recycling prevents unbounded memory growth across many trials in a long
    grid.  On HPC the RAM headroom is huge, so we recycle rarely (less cold-
    start overhead = faster grid).  On laptops we recycle aggressively.
    """
    return {"hpc": 64, "workstation": 16, "laptop": 4}.get(env, 8)


def _apply_env_flags(env: str, trials: list[dict[str, Any]]) -> None:
    """Stamp torch_compile / torch_compile_mode onto every trial's hp in place."""
    for t in trials:
        compile_on, compile_mode = _compile_flags(env, t["arch"])
        t["hp"]["torch_compile"]      = compile_on
        t["hp"]["torch_compile_mode"] = compile_mode


# ============================================================================
# ── SEQUENTIAL RUNNER ────────────────────────────────────────────────────────
# ============================================================================

def _wait_for_memory(log: logging.Logger) -> None:
    import psutil
    if psutil.virtual_memory().available / 1e9 >= MIN_FREE_RAM_GB:
        return
    log.warning(
        "Only %.1f GB RAM free — pausing %.0f s ...",
        psutil.virtual_memory().available / 1e9, MEM_POLL_INTERVAL,
    )
    while psutil.virtual_memory().available / 1e9 < MIN_FREE_RAM_GB:
        time.sleep(MEM_POLL_INTERVAL)


def _run_sequential(
    trials: list[dict[str, Any]],
    log: logging.Logger,
    t_start: float,
) -> tuple[int, int, int]:
    """Run every trial in-process, one after another (CPU-only or pool_size=1)."""
    import psutil
    import torch

    env = _detect_env()
    _apply_env_flags(env, trials)

    # Thread setup — main process can use all cores now (no siblings).
    cpu_phys = psutil.cpu_count(logical=False) or 2
    torch_threads = max(1, cpu_phys - 1)
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(max(1, torch_threads // 4))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.cuda.set_device(0)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")  # expose both GPUs; pipeline uses cuda:0
        if env == "hpc":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32        = True
    # Sequential main process IS NOT daemonic, so DataLoader worker subprocesses
    # are fine here. Pipeline's own auto-cap still applies (RAM-based ceiling).
    _seq_workers = str(min(4, max(1, (os.cpu_count() or 2) // 2)))
    os.environ["NN_NUM_WORKERS"] = _seq_workers
    os.environ["_GRID_N_WORKERS"] = "1"

    logging.getLogger("Neural_Networks.models.shared.pipeline").setLevel(logging.WARNING)
    logging.getLogger("Neural_Networks").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    from Neural_Networks.models.shared.pipeline import TrainJob, run_training
    from Neural_Networks.models.shared.strategies import (
        PLAIN_STRATEGY, PHYSICS_REG_STRATEGY, RESIDUAL_STRATEGY,
    )
    strategy_map = {"fnn": PLAIN_STRATEGY, "physreg": PHYSICS_REG_STRATEGY, "residual": RESIDUAL_STRATEGY}

    pending, n_skip = _partition_trials_by_skip(trials)
    if n_skip:
        log.info("Pre-dispatch SKIP: %d/%d trials already complete in this batch.",
                 n_skip, len(trials))
    total = len(pending)
    n_ok = n_fail = 0
    w = len(str(max(1, total)))

    for idx, trial in enumerate(pending, 1):
        arch        = trial["arch"]
        model_type  = trial["model_type"]
        save_subdir = trial["save_subdir"]
        run_help    = trial["run_help"]
        hp          = dict(trial["hp"])
        models_dir  = os.path.join(DATASET_OUT_ROOT, save_subdir)
        os.makedirs(models_dir, exist_ok=True)
        desc = _hp_desc(arch, hp)

        _wait_for_memory(log)

        # Belt-and-suspenders: pre-filter runs before the loop, but a parallel
        # rerun of the same batch could complete a twin in-flight during the loop.
        if SKIP_EXISTING and _find_existing_run(Path(models_dir), model_type, hp):
            n_skip += 1
            tqdm.write(f"  [SKIP]  {idx:{w}}/{total}  {desc}")
            continue

        # torch_compile / torch_compile_mode already set by _apply_env_flags().

        epochs_total = int(hp.get("epochs", 3000))
        bar = tqdm(
            total=epochs_total,
            desc=f"[{idx:{w}}/{total}] {desc}",
            unit="ep",
            leave=True,
            dynamic_ncols=True,
            bar_format="{desc}  {bar}  {n_fmt}/{total_fmt} ep  {postfix}",
        )
        last_epoch = [0]

        def _cb(
            epoch: int, total_ep: int, val_rmse: float, pat_ctr: int, pat_max: int,
            _bar: tqdm = bar, _last: list = last_epoch,
        ) -> None:
            delta = epoch - _last[0]
            if delta > 0:
                _bar.update(delta)
                _last[0] = epoch
            _bar.set_postfix_str(
                f"val_rmse={val_rmse:.4f} N·m  pat={pat_ctr}/{pat_max}",
                refresh=True,
            )

        job = TrainJob(
            run_dir=TRAIN_DATA_RUN_DIR,
            models_dir=models_dir,
            registry_file=REGISTRY_FILE,
            model_type=model_type,
            save_subdir=save_subdir,
            hp=hp,
            strategy=strategy_map[arch],
            run_help=run_help,
        )

        status    = "error"
        error_msg = ""
        try:
            rc     = run_training(job, progress_callback=_cb)
            status = "ok" if rc == 0 else "failed"
        except Exception as exc:
            import traceback
            error_msg = f"{exc}\n{traceback.format_exc()}"
            status    = "error"
        finally:
            bar.close()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

        elapsed = time.time() - t_start
        eta     = (elapsed / idx) * (total - idx) if idx > 0 else 0.0
        if status == "ok":
            n_ok += 1
            tqdm.write(f"  [ OK ]  {idx:{w}}/{total}  {desc}  elapsed={_fmt_time(elapsed)}  ETA={_fmt_time(eta)}")
        elif status == "failed":
            n_fail += 1
            tqdm.write(f"  [FAIL]  {idx:{w}}/{total}  {desc}  (run_training returned non-zero)")
        else:
            n_fail += 1
            tqdm.write(f"  [ERR ]  {idx:{w}}/{total}  {desc}\n" + error_msg[:400])

    return n_ok, n_skip, n_fail


# ============================================================================
# ── PARALLEL WORKER ──────────────────────────────────────────────────────────
#
# Under the *spawn* start method the module is re-imported in each child, so
# CUDA-heavy libs MUST NOT be imported at module top.  Queues cannot be passed
# as ``apply_async`` args under spawn (pickle restriction), so the Pool
# initializer caches the shared progress queue in a module global.
# ============================================================================

_POOL_PROGRESS_QUEUE: "mp.Queue | None" = None
_POOL_THREADS_PER_WORKER: int = 2
_POOL_GPU_ID: int = 0


def _pool_init(
    progress_queue: "mp.Queue",
    threads_per_worker: int,
    gpu_ticket_q: "mp.Queue",
    is_hpc: bool = False,
) -> None:
    """Pool initializer: runs ONCE per worker lifetime (on spawn and after each
    ``maxtasksperchild`` recycle).

    Critically, ``torch.set_num_interop_threads`` can only be called before any
    parallel work has started.  Calling it per-trial crashed every trial after
    the first one in each worker.  Doing it here — exactly once — keeps the
    worker usable for its full ``maxtasksperchild`` budget.
    """
    global _POOL_PROGRESS_QUEUE, _POOL_THREADS_PER_WORKER, _POOL_GPU_ID
    _POOL_PROGRESS_QUEUE = progress_queue
    _POOL_THREADS_PER_WORKER = int(threads_per_worker)
    # Each worker pops its unique GPU assignment from the ticket queue.
    # Workers are distributed round-robin: slot 0→GPU 0, slot 1→GPU 1, etc.
    _POOL_GPU_ID = int(gpu_ticket_q.get())

    # Silence worker-process logging noise.
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    for name in (
        "", "Neural_Networks", "Neural_Networks.loader",
        "Neural_Networks.models.shared.pipeline", "Neural_Networks.robot_physics",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Env vars must be set before torch import. Main process does not touch
    # CUDA so the worker inherits a clean slate.
    os.environ["NN_NUM_WORKERS"] = "0"      # daemonic workers can't spawn children
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_POOL_GPU_ID)

    import torch
    torch.set_num_threads(int(threads_per_worker))
    # This call is one-shot per process — must not repeat in _run_one_trial.
    try:
        torch.set_num_interop_threads(max(1, int(threads_per_worker) // 2))
    except RuntimeError:
        # If a prior import already kicked off parallel work, swallow quietly.
        pass
    if torch.cuda.is_available():
        # TF32: ~3x faster float32 matmul on Ampere+ (A100/H100) with negligible
        # precision loss for ML training.  set_float32_matmul_precision("high")
        # enables both matmul and cuDNN paths.
        torch.set_float32_matmul_precision("high")
        if is_hpc:
            # Explicit TF32 flags in case an older torch ignores the above.
            torch.backends.cuda.matmul.allow_tf32  = True
            torch.backends.cudnn.allow_tf32        = True


def _run_one_trial(
    trial: dict[str, Any],
    worker_slot: int,
    threads_per_worker: int,
) -> dict[str, Any]:
    """Run a single trial in a fresh worker process.

    Must set ``CUDA_VISIBLE_DEVICES`` and thread counts BEFORE importing torch,
    so this function does its imports lazily inside the body.
    """
    q = _POOL_PROGRESS_QUEUE
    if q is None:
        raise RuntimeError("_pool_init did not run — progress queue missing")

    arch        = trial["arch"]
    model_type  = trial["model_type"]
    save_subdir = trial["save_subdir"]
    run_help    = trial["run_help"]
    hp          = dict(trial["hp"])
    models_dir  = os.path.join(DATASET_OUT_ROOT, save_subdir)
    os.makedirs(models_dir, exist_ok=True)

    # Thread counts, CUDA_VISIBLE_DEVICES, and NN_NUM_WORKERS are all set
    # exactly once per worker in ``_pool_init``.  torch.set_num_interop_threads
    # is a one-shot call — repeating it here on task 2+ raised RuntimeError
    # ("cannot set number of interop threads after parallel work has started")
    # which manifested as "ERR after 1 s" for every trial except the first.

    try:
        import torch

        # Skip-existing inside the worker — cheap, avoids redundant data load.
        if SKIP_EXISTING and _find_existing_run(Path(models_dir), model_type, hp):
            q.put(("done", worker_slot, "skip"))
            return {"status": "skipped", "arch": arch, "hp": hp}

        # torch_compile / torch_compile_mode already set by _apply_env_flags()
        # in the main process before dispatch — no override here.

        # Late imports: don't touch CUDA libs at module level.
        from Neural_Networks.models.shared.pipeline import TrainJob, run_training
        from Neural_Networks.models.shared.strategies import (
            PLAIN_STRATEGY, PHYSICS_REG_STRATEGY, RESIDUAL_STRATEGY,
        )
        strategy_map = {"fnn": PLAIN_STRATEGY, "physreg": PHYSICS_REG_STRATEGY, "residual": RESIDUAL_STRATEGY}

        job = TrainJob(
            run_dir=TRAIN_DATA_RUN_DIR,
            models_dir=models_dir,
            registry_file=REGISTRY_FILE,
            model_type=model_type,
            save_subdir=save_subdir,
            hp=hp,
            strategy=strategy_map[arch],
            run_help=run_help,
        )

        q.put(("start", worker_slot, _hp_desc(arch, hp),
               int(hp.get("epochs", 3000)), int(hp.get("patience", 200))))

        def _cb(epoch: int, total_ep: int, val_rmse: float, pat_ctr: int, pat_max: int) -> None:
            q.put(("progress", worker_slot, int(epoch), int(total_ep),
                   float(val_rmse), int(pat_ctr), int(pat_max)))

        rc = run_training(job, progress_callback=_cb)
        status = "ok" if rc == 0 else "failed"
        q.put(("done", worker_slot, status))
        return {"status": status, "arch": arch, "hp": hp}

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        if _POOL_PROGRESS_QUEUE is not None:
            _POOL_PROGRESS_QUEUE.put(("done", worker_slot, "error"))
        return {"status": "error", "arch": arch, "hp": hp, "error": f"{exc}\n{tb}"}

    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


# ============================================================================
# ── TUI STATE + DASHBOARD (rich-based) ───────────────────────────────────────
# ============================================================================

@dataclass
class _SlotState:
    slot: int
    hp_desc: str = ""
    arch: str = ""
    epoch: int = 0
    total_ep: int = 1
    val_rmse: float = float("nan")
    pat_ctr: int = 0
    pat_max: int = 0   # filled from hp["patience"] in "start" message
    started_at: float | None = None
    status: str = "waiting"   # waiting | running | done


@dataclass
class _Result:
    n: int
    arch: str
    config: str
    status: str               # ok | skip | fail | err
    rmse: float | None
    elapsed: float


@dataclass
class _TUIState:
    pool_size: int
    total_trials: int
    threads_per_worker: int
    device_label: str
    slots: list[_SlotState] = field(default_factory=list)
    in_flight: int = 0
    pending: int = 0
    completed: int = 0
    ok: int = 0
    skip: int = 0
    fail: int = 0
    free_vram_gb:  float = 0.0
    total_vram_gb: float = 0.0
    free_ram_gb:   float = 0.0
    total_ram_gb:  float = 0.0
    results: deque = field(default_factory=lambda: deque(maxlen=8))
    t_start: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


_BLOCK_CHARS = " ▏▎▍▌▋▊▉█"


def _gauge(used: float, total: float, width: int = 24) -> str:
    """Render a fractional unicode-block gauge."""
    if total <= 0:
        return " " * width
    frac = max(0.0, min(1.0, used / total))
    units = frac * width
    full = int(units)
    part_idx = int((units - full) * (len(_BLOCK_CHARS) - 1))
    part = _BLOCK_CHARS[part_idx] if full < width else ""
    return "█" * full + part + " " * max(0, width - full - (1 if part else 0))


def _fmt_hms(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec//3600:d}:{(sec%3600)//60:02d}:{sec%60:02d}"


def _fmt_mmss(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec//60:02d}:{sec%60:02d}"


def _short_config(hp: dict) -> str:
    """Compact HP string for dashboard tables."""
    frac = hp.get("data_train_fraction", "?")
    seed = hp.get("seed", "?")
    extra = ""
    if "physics_weight" in hp:
        extra = f" pw={hp['physics_weight']}"
    elif "alpha_reg_weight" in hp:
        extra = f" arw={hp['alpha_reg_weight']}"
    return f"frac={frac} seed={seed}{extra}"


def _use_rich() -> bool:
    """Whether to render the live dashboard (TTY only)."""
    try:
        return bool(sys.stdout.isatty() and sys.stderr.isatty()
                    and os.environ.get("TERM") not in ("", "dumb"))
    except Exception:
        return False


def _render_dashboard(state: _TUIState) -> Any:
    """Build the rich Layout for the current TUI state.  Reads under lock."""
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align

    with state.lock:
        # Snapshot every field we need so rendering is lock-free after here.
        pool_size   = state.pool_size
        total       = state.total_trials
        tpw         = state.threads_per_worker
        device      = state.device_label
        in_flight   = state.in_flight
        pending     = state.pending
        completed   = state.completed
        ok_n        = state.ok
        skip_n      = state.skip
        fail_n      = state.fail
        free_vram   = state.free_vram_gb
        total_vram  = state.total_vram_gb
        free_ram    = state.free_ram_gb
        total_ram   = state.total_ram_gb
        slots       = [
            _SlotState(
                slot=s.slot, hp_desc=s.hp_desc, arch=s.arch,
                epoch=s.epoch, total_ep=s.total_ep, val_rmse=s.val_rmse,
                pat_ctr=s.pat_ctr, pat_max=s.pat_max,
                started_at=s.started_at, status=s.status,
            )
            for s in state.slots
        ]
        results     = list(state.results)
        t_start     = state.t_start

    # ── Header ──────────────────────────────────────────────────────────────
    header = Text.assemble(
        ("Torque Grid Search", "bold cyan"),
        "   ARCH=", (str(ARCH), "bold"),
        f"   total={total}   ",
        (f"pool_size={pool_size}", "bold"),
        f"   threads/worker={tpw}   device={device}",
    )
    header_panel = Panel(header, border_style="cyan", padding=(0, 1))

    # ── Resources ───────────────────────────────────────────────────────────
    vram_used = max(0.0, total_vram - free_vram)
    ram_used  = max(0.0, total_ram  - free_ram)
    vram_pct  = (vram_used / total_vram * 100) if total_vram > 0 else 0.0
    ram_pct   = (ram_used  / total_ram  * 100) if total_ram  > 0 else 0.0

    def _gauge_style(pct: float) -> str:
        if pct >= 85: return "bold red"
        if pct >= 65: return "yellow"
        return "green"

    res_tbl = Table.grid(padding=(0, 1), expand=True)
    res_tbl.add_column(style="dim", width=6, no_wrap=True)
    res_tbl.add_column(width=26, no_wrap=True)
    res_tbl.add_column(no_wrap=True)
    res_tbl.add_row(
        "VRAM",
        Text(_gauge(vram_used, total_vram, 24), style=_gauge_style(vram_pct)),
        f"{vram_used:5.2f} / {total_vram:5.2f} GB  ({free_vram:4.2f} free, reserve {VRAM_RESERVE_GB:.2f})",
    )
    res_tbl.add_row(
        "RAM",
        Text(_gauge(ram_used, total_ram, 24), style=_gauge_style(ram_pct)),
        f"{ram_used:5.2f} / {total_ram:5.2f} GB  ({free_ram:4.2f} free, reserve {RAM_RESERVE_GB:.2f})",
    )

    counts = Text.assemble(
        f"in-flight ",
        (f"{in_flight:>2d}", "bold cyan"),
        "     pending ",
        (f"{pending:>3d}", "bold"),
        "     completed ",
        (f"{completed:>3d}", "bold"),
        "     ",
        ("OK ",   "bold"), (f"{ok_n:>2d}",   "bold green"),
        "   ", ("SKIP ", "bold"), (f"{skip_n:>2d}", "bold yellow"),
        "   ", ("FAIL ", "bold"), (f"{fail_n:>2d}", "bold red"),
    )

    res_outer = Table.grid(expand=True)
    res_outer.add_row(res_tbl)
    res_outer.add_row(counts)
    resources_panel = Panel(res_outer, title="resources (live)", border_style="blue", padding=(0, 1))

    # ── Active trials (two lines per slot, capped at 5 visible) ────────────
    # Sort: running slots first, then waiting — so the most informative rows
    # are always visible when pool_size > 5.
    _TUI_MAX_SLOTS = 5
    slots_sorted = sorted(slots, key=lambda s: (0 if s.status == "running" else 1, s.slot))
    slots_visible = slots_sorted[:_TUI_MAX_SLOTS]
    hidden = len(slots) - len(slots_visible)

    act = Table.grid(padding=(0, 0), expand=True)
    act.add_column()

    now = time.time()
    first = True
    for s in slots_visible:
        if not first:
            act.add_row("")   # visual separator between slots
        first = False

        tag = f"slot {s.slot+1}/{pool_size}"
        if s.status == "running":
            bar = _gauge(s.epoch, max(1, s.total_ep), 28)
            elapsed = now - s.started_at if s.started_at else 0.0
            eta = (elapsed / s.epoch) * (s.total_ep - s.epoch) if s.epoch > 0 else 0.0
            rmse_str = f"{s.val_rmse:.4f}" if not math.isnan(s.val_rmse) else "  —"
            # Strip the arch prefix from hp_desc — we render arch in its own column.
            cfg = s.hp_desc
            if s.arch and cfg.lstrip().startswith(s.arch):
                cfg = cfg.lstrip()[len(s.arch):].lstrip()
            line1 = Text.assemble(
                (f"{tag}  ", "dim"),
                (f"{s.arch:<9}", "cyan"),
                f"{cfg[:32]:<32}  ",
                (bar, "cyan"),
                f"  {s.epoch:>4d}/{s.total_ep:<4d}",
            )
            line2 = Text.assemble(
                "            ",
                ("val_rmse=", "dim"), f"{rmse_str}  ",
                ("pat=",      "dim"), f"{s.pat_ctr:>2d}/{s.pat_max:<2d}  ",
                ("elapsed ",  "dim"), _fmt_mmss(elapsed),
                ("   eta ",   "dim"), _fmt_mmss(eta),
            )
            act.add_row(line1)
            act.add_row(line2)
        else:
            line1 = Text.assemble((f"{tag}  ", "dim"), ("waiting ...", "dim"))
            act.add_row(line1)
            act.add_row(Text("", style="dim"))
    if hidden > 0:
        act.add_row(Text(f"  … {hidden} more slot(s) not shown (pool_size={pool_size})", style="dim"))
    active_panel = Panel(act, title=f"active trials (showing {len(slots_visible)}/{pool_size})", border_style="cyan", padding=(0, 1))

    # ── Overall progress ────────────────────────────────────────────────────
    overall_bar_w = 40
    overall_bar = _gauge(completed, max(1, total), overall_bar_w)
    elapsed_s = now - t_start
    avg_per = elapsed_s / completed if completed > 0 else 0.0
    eta_s = avg_per * (total - completed) if completed > 0 else 0.0
    tp_str = f"{avg_per/60:.2f} min/trial" if completed > 0 else "—"
    overall_inner = Table.grid(expand=True)
    overall_inner.add_column(width=overall_bar_w + 2, no_wrap=True)
    overall_inner.add_column(no_wrap=True)
    overall_inner.add_row(
        Text(overall_bar, style="bold green"),
        f"  {completed} / {total} trials",
    )
    overall_inner.add_row(
        f"  elapsed {_fmt_hms(elapsed_s):>8s}    eta {_fmt_hms(eta_s):>8s}    throughput {tp_str}",
        "",
    )
    overall_panel = Panel(overall_inner, title="overall progress", border_style="green", padding=(0, 1))

    # ── Recent results ──────────────────────────────────────────────────────
    hist = Table(expand=True, show_edge=False, pad_edge=False, padding=(0, 1))
    hist.add_column("#",        width=4,  justify="right", style="dim")
    hist.add_column("arch",     width=9,  no_wrap=True)
    hist.add_column("config",   width=28, no_wrap=True)
    hist.add_column("result",   width=6,  justify="center")
    hist.add_column("rmse",     width=8,  justify="right")
    hist.add_column("elapsed",  justify="right")
    if not results:
        hist.add_row("—", "", "", "", "", "")
    for r in reversed(results):
        style = {"ok": "green", "skip": "yellow", "fail": "red", "err": "red"}.get(r.status, "")
        label = {"ok": "OK", "skip": "SKIP", "fail": "FAIL", "err": "ERR"}.get(r.status, r.status.upper())
        hist.add_row(
            str(r.n),
            Text(r.arch, style="cyan"),
            r.config,
            Text(label, style=style),
            f"{r.rmse:.4f}" if r.rmse is not None else "—",
            _fmt_hms(r.elapsed),
        )
    history_panel = Panel(hist, title="recent results (last 8)", border_style="magenta", padding=(0, 1))

    # ── Assemble layout ─────────────────────────────────────────────────────
    layout = Layout()
    # 2 content lines per visible slot + 1 separator + 1 "…N more" line if hidden.
    n_vis = len(slots_visible)
    active_h = 2 + 2 * n_vis + max(0, n_vis - 1) + (1 if hidden > 0 else 0)
    history_h = 3 + max(len(results), 1)        # chrome + header + rows
    layout.split_column(
        Layout(header_panel,    name="header",    size=3),
        Layout(resources_panel, name="resources", size=6),
        Layout(active_panel,    name="active",    size=active_h),
        Layout(overall_panel,   name="overall",   size=5),
        Layout(history_panel,   name="history",   size=min(12, history_h)),
    )
    return layout


class _DashboardRenderable:
    """rich ``__rich__`` hook so ``Live`` auto-refresh pulls live state."""
    def __init__(self, state: _TUIState):
        self.state = state

    def __rich__(self) -> Any:
        return _render_dashboard(self.state)


# ============================================================================
# ── PARALLEL RUNNER (dynamic admission, rich dashboard) ──────────────────────
# ============================================================================

def _run_parallel_dynamic(
    trials: list[dict[str, Any]],
    pool_size: int,
    threads_per_worker: int,
    device_label: str,
    log: logging.Logger,
    t_start: float,
) -> tuple[int, int, int]:
    """Submit trials while live free VRAM + RAM permit.  No hardcoded N.

    Renders a live rich dashboard on TTY; falls back to periodic plain status
    lines when stdout/stderr are redirected.
    """
    # Detect HPC/workstation/laptop once and stamp compile flags on every trial.
    env = _detect_env()
    _apply_env_flags(env, trials)
    is_hpc = env == "hpc"

    original_total = len(trials)
    trials, n_pre_skip = _partition_trials_by_skip(trials)
    if n_pre_skip:
        log.info("Pre-dispatch SKIP: %d/%d trials already complete in this batch.",
                 n_pre_skip, original_total)
    if not trials:
        log.info("Nothing to do — all %d trials already complete.", original_total)
        return 0, n_pre_skip, 0
    total = len(trials)

    # One-time measurement of the per-process CUDA context overhead.
    cuda_ctx_gb = _measure_cuda_ctx_gb()
    log.info("CUDA ctx overhead (per worker): %.2f GB", cuda_ctx_gb)

    # Per-trial estimates — different HPs will differ in future grids.
    estimates = [_estimate_trial_mem(t["hp"], t["arch"], cuda_ctx_gb) for t in trials]
    vram_max = max(e[0] for e in estimates)
    vram_min = min(e[0] for e in estimates)
    ram_max  = max(e[1] for e in estimates)
    log.info("Per-trial estimate: vram∈[%.2f, %.2f] GB  ram_max=%.2f GB", vram_min, vram_max, ram_max)

    # Largest-first: big trials slot in when VRAM is cold.
    pending: list[tuple[dict, tuple[float, float]]] = sorted(
        zip(trials, estimates), key=lambda x: -x[1][0]
    )

    # ── Shared dashboard state ──────────────────────────────────────────────
    import psutil as _ps
    _total_ram_gb = _ps.virtual_memory().total / 1e9
    state = _TUIState(
        pool_size=pool_size,
        total_trials=total,
        threads_per_worker=threads_per_worker,
        device_label=device_label,
        slots=[_SlotState(slot=i) for i in range(pool_size)],
        pending=total,
        total_vram_gb=_query_total_vram_gb() or 0.0,
        total_ram_gb=_total_ram_gb,
        t_start=t_start,
    )
    state.free_vram_gb = _query_free_vram_gb()
    state.free_ram_gb  = _query_free_ram_gb()

    ctx = mp.get_context("spawn")
    progress_q: mp.Queue = ctx.Queue()

    # ── Drain thread: queue messages → dashboard state ──────────────────────
    stop_event = threading.Event()

    def _drain() -> None:
        while not stop_event.is_set():
            try:
                msg = progress_q.get(timeout=0.5)
            except Exception:
                continue
            if msg is None:
                return
            kind, slot = msg[0], int(msg[1])
            if not (0 <= slot < pool_size):
                continue
            with state.lock:
                s = state.slots[slot]
                if kind == "start":
                    _, _, desc, epochs_total, patience = msg
                    s.hp_desc    = str(desc)
                    s.arch       = str(desc).split()[0] if desc else ""
                    s.epoch      = 0
                    s.total_ep   = max(1, int(epochs_total))
                    s.val_rmse   = float("nan")
                    s.pat_ctr    = 0
                    s.pat_max    = int(patience)   # actual patience from hp, not hardcoded
                    s.started_at = time.time()
                    s.status     = "running"
                elif kind == "progress":
                    _, _, epoch, total_ep, val_rmse, pat_ctr, pat_max = msg
                    s.epoch    = int(epoch)
                    s.total_ep = max(1, int(total_ep))
                    s.val_rmse = float(val_rmse)
                    s.pat_ctr  = int(pat_ctr)
                    s.pat_max  = int(pat_max)
                elif kind == "done":
                    s.status = "waiting"   # result row added by main-thread reaper

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()

    # ── Resource poller: live VRAM/RAM gauges at 1 Hz ───────────────────────
    def _poll_resources() -> None:
        while not stop_event.is_set():
            fv = _query_free_vram_gb()
            fr = _query_free_ram_gb()
            with state.lock:
                state.free_vram_gb = fv
                state.free_ram_gb  = fr
            stop_event.wait(1.0)

    poller_thread = threading.Thread(target=_poll_resources, daemon=True)
    poller_thread.start()

    # ── Fallback ticker (non-TTY): periodic plain status line ───────────────
    use_rich_ui = _use_rich()

    def _fallback_ticker() -> None:
        while not stop_event.is_set():
            with state.lock:
                msg = (
                    f"[{_fmt_hms(time.time() - state.t_start)}] "
                    f"done={state.completed}/{state.total_trials}  "
                    f"in-flight={state.in_flight}  pending={state.pending}  "
                    f"vram={max(0.0, state.total_vram_gb - state.free_vram_gb):.2f}/"
                    f"{state.total_vram_gb:.2f} GB  "
                    f"ram={max(0.0, state.total_ram_gb - state.free_ram_gb):.2f}/"
                    f"{state.total_ram_gb:.2f} GB  "
                    f"ok={state.ok} skip={state.skip} fail={state.fail}"
                )
            print(msg, flush=True)
            stop_event.wait(10.0)

    fallback_thread: threading.Thread | None = None
    if not use_rich_ui:
        fallback_thread = threading.Thread(target=_fallback_ticker, daemon=True)
        fallback_thread.start()

    # ── Admission loop primitives ───────────────────────────────────────────
    in_flight_map: dict[Any, tuple[dict, tuple[float, float], int, float]] = {}
    free_slots: list[int] = list(range(pool_size))

    def _set_counts() -> None:
        with state.lock:
            state.in_flight = len(in_flight_map)
            state.pending   = len(pending)

    def _reap_completed() -> None:
        for ar in [a for a in list(in_flight_map) if a.ready()]:
            trial, _est, slot, submitted_at = in_flight_map.pop(ar)
            free_slots.append(slot)
            try:
                result = ar.get(timeout=1.0)
            except Exception as exc:
                result = {"status": "error", "arch": trial["arch"], "hp": trial["hp"], "error": str(exc)}
            status = result.get("status", "?")
            # Normalise status into dashboard codes
            code = {
                "ok": "ok", "skipped": "skip", "failed": "fail", "error": "err",
            }.get(status, "err")
            # Surface the traceback for failed trials. Previously the dashboard
            # swallowed the result["error"] string and only showed "ERR".
            if code in ("err", "fail"):
                err_msg = str(result.get("error", "") or "").strip()
                _errs_dir = Path(DATASET_OUT_ROOT) / "_errors"
                _errs_dir.mkdir(parents=True, exist_ok=True)
                _log_path = _errs_dir / "trial_errors.log"
                with open(_log_path, "a") as fh:
                    fh.write("=" * 72 + "\n")
                    fh.write(f"[{datetime.now().isoformat()}]  "
                             f"arch={trial['arch']}  {_hp_desc(trial['arch'], trial['hp'])}\n")
                    fh.write(err_msg[:4000] + "\n")
                # Also log a one-line summary so headless runs still see it.
                first_line = err_msg.splitlines()[-1] if err_msg else "(no traceback)"
                log.warning("Trial FAIL  %s  →  %s  (see %s)",
                            _hp_desc(trial['arch'], trial['hp']), first_line[:160], _log_path)
            elapsed_trial = time.time() - submitted_at
            rmse = None  # rmse for the run is not returned via result dict; we surface "—"
            with state.lock:
                state.completed += 1
                if code == "ok":   state.ok   += 1
                elif code == "skip": state.skip += 1
                else:              state.fail += 1
                state.results.append(_Result(
                    n=state.completed,
                    arch=trial["arch"],
                    config=_short_config(trial["hp"]),
                    status=code,
                    rmse=rmse,
                    elapsed=elapsed_trial,
                ))
                # Reset the slot that just freed up
                if 0 <= slot < pool_size:
                    st = state.slots[slot]
                    st.status = "waiting"
                    st.hp_desc = ""
                    st.arch = ""
                    st.epoch = 0
                    st.total_ep = 1
                    st.val_rmse = float("nan")
                    st.pat_ctr = 0
                    st.started_at = None
            _set_counts()

    def _try_admit_one(pool) -> bool:
        """Dispatch the next trial if resources permit.

        VRAM is a hard gate (GPU OOM crashes the trial outright).  RAM is an
        advisory signal only: once pool workers are spawned their base RSS is
        already charged against ``ram_used``, and every admission targets a
        reused worker whose marginal cost is just the trial's tensor working
        set.  Gating dispatch on full RAM estimate created a deadlock — idle
        workers couldn't take new work, and idle workers don't release RAM,
        so the stall was permanent.  Now we only log the tightness.
        """
        if not pending or not free_slots:
            return False
        trial, (vram_est, ram_est) = pending[0]      # largest-first
        with state.lock:
            vram_free = state.free_vram_gb
            ram_free  = state.free_ram_gb
        vram_ok = vram_free >= vram_est + VRAM_RESERVE_GB
        if vram_ok:
            if ram_free < RAM_RESERVE_GB:
                log.warning(
                    "RAM tight (free=%.2f, reserve=%.2f) — dispatching anyway; "
                    "reused workers only pay marginal tensor cost.",
                    ram_free, RAM_RESERVE_GB,
                )
            pending.pop(0)
            slot = free_slots.pop(0)
            ar = pool.apply_async(_run_one_trial, (trial, slot, threads_per_worker))
            in_flight_map[ar] = (trial, (vram_est, ram_est), slot, time.time())
            _set_counts()
            return True
        # VRAM didn't fit for largest.  If pool is empty, try smallest.
        if not in_flight_map:
            pending.sort(key=lambda x: x[1][0])
            trial, (vram_est, ram_est) = pending[0]
            if vram_free >= vram_est + VRAM_RESERVE_GB:
                pending.pop(0)
                slot = free_slots.pop(0)
                ar = pool.apply_async(_run_one_trial, (trial, slot, threads_per_worker))
                in_flight_map[ar] = (trial, (vram_est, ram_est), slot, time.time())
                pending.sort(key=lambda x: -x[1][0])
                _set_counts()
                return True
            pending.sort(key=lambda x: -x[1][0])
            log.warning(
                "VRAM tight — free=%.2f GB; smallest trial needs %.2f GB. sleeping %.1fs",
                vram_free, vram_est, TIGHT_SLEEP_SEC,
            )
            time.sleep(TIGHT_SLEEP_SEC)
        return False

    _set_counts()

    # ── Pool + dashboard lifecycle ──────────────────────────────────────────
    maxtasks = _maxtasks_for_env(env)   # hpc=64, workstation=16, laptop=4
    # Build a GPU ticket queue: each worker pops its GPU ID on init.
    # Workers are distributed round-robin across available GPUs.
    n_gpus = _query_n_gpus()
    gpu_ticket_q: mp.Queue = ctx.Queue()
    for i in range(pool_size):
        gpu_ticket_q.put(i % n_gpus)
    pool = ctx.Pool(
        processes=pool_size,
        maxtasksperchild=maxtasks,
        initializer=_pool_init,
        initargs=(progress_q, threads_per_worker, gpu_ticket_q, is_hpc),
    )

    live_cm = None
    if use_rich_ui:
        from rich.console import Console
        from rich.live import Live
        console = Console()
        live_cm = Live(
            _DashboardRenderable(state),
            console=console,
            refresh_per_second=4,
            screen=False,
            transient=False,
        )
        live_cm.__enter__()

    try:
        # Initial fill
        while pending and free_slots:
            if not _try_admit_one(pool):
                break

        # Main admission + reap loop
        while pending or in_flight_map:
            _reap_completed()
            if pending and free_slots:
                admitted = _try_admit_one(pool)
                if not admitted and in_flight_map:
                    time.sleep(ADMISSION_POLL_SEC)
            elif in_flight_map:
                time.sleep(ADMISSION_POLL_SEC)

    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt — terminating worker pool ...")
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()
    finally:
        stop_event.set()
        progress_q.put(None)
        drain_thread.join(timeout=3)
        poller_thread.join(timeout=3)
        if fallback_thread is not None:
            fallback_thread.join(timeout=3)
        if live_cm is not None:
            try:
                live_cm.__exit__(None, None, None)
            except Exception:
                pass

    with state.lock:
        return state.ok, state.skip + n_pre_skip, state.fail


# ============================================================================
# ── MAIN ─────────────────────────────────────────────────────────────────────
# ============================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    log = logging.getLogger("grid")

    trials      = _build_trials()
    total       = len(trials)
    arch_counts = Counter(t["arch"] for t in trials)

    log.info("=" * 72)
    log.info("Torque Model  —  Local Grid Search (dynamic admission)")
    log.info("  ARCH       : %s", ARCH)
    log.info("  TOTAL      : %d trials", total)
    for a, cnt in sorted(arch_counts.items()):
        log.info("    %-10s  %d combos", a, cnt)
    log.info("  DATA       : %s", TRAIN_DATA_RUN_DIR)
    log.info("  DATASET    : %s", _DATASET_NAME)
    log.info("  OUTPUT     : %s", DATASET_OUT_ROOT)
    log.info("  SKIP_EXIST : %s  (pre-dispatch)", SKIP_EXISTING)
    log.info("  SPEED-UPS  : memmap=%s, max_epochs=%d, patience=%d",
             FIXED_HP.get("dataset_memmap", False),
             FIXED_HP.get("epochs", 3000),
             FIXED_HP.get("patience", 200))
    log.info("               pool_size=min(floor(vram_total/0.8), cpu_phys-2)")
    log.info("               maxtasksperchild: hpc=64  workstation=16  laptop=4")
    log.info("               torch.compile:    hpc=reduce-overhead  workstation=default  laptop=off")
    log.info("  FIXES      : PhysReg cal frozen (l_phys anchored to sum(phys));")
    log.info("               Residual α∈{0,0.1,1.0}, final-layer init×1e-2")
    log.info("  GRID       : 3 fracs × {BlackBox, PhysReg{0,0.3,1.0}, Residual{0,0.1,1.0}}")
    log.info("               × 2 seeds = 42 trials (minimal, converged via patience=%d)",
             FIXED_HP.get("patience", 200))
    log.info("=" * 72)

    if DRY_RUN:
        log.info("DRY_RUN=True — printing combo table and exiting.")
        _print_combo_table(trials)
        return

    for arch in _ARCH_META:
        _, save_subdir, _ = _ARCH_META[arch]
        os.makedirs(os.path.join(DATASET_OUT_ROOT, save_subdir), exist_ok=True)
    os.makedirs(os.path.join(DATASET_OUT_ROOT, "analysis"), exist_ok=True)

    # Probe capabilities (no CUDA context created in the main process).
    cuda_ok = _query_cuda_available()
    pool_size = _compute_pool_size(cuda_ok)
    threads_per_worker = _compute_threads_per_worker(pool_size)

    import psutil
    cpu_phys = psutil.cpu_count(logical=False) or 2
    free_ram_gb = _query_free_ram_gb()
    if cuda_ok:
        free_vram_gb  = _query_free_vram_gb()
        total_vram_gb = _query_total_vram_gb()
        # Short device label for the dashboard header.
        try:
            import subprocess as _sp
            gpu_name = _sp.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                timeout=5,
            ).decode().strip().split("\n")[0]
        except Exception:
            gpu_name = "CUDA GPU"
        device_label = f"cuda:0 ({gpu_name}, {total_vram_gb:.1f} GB)"
        log.info(
            "device=%s  VRAM=%.1f/%.1f GB free   RAM=%.1f GB free   CPU=%d phys",
            device_label, free_vram_gb, total_vram_gb, free_ram_gb, cpu_phys,
        )
    else:
        device_label = "cpu"
        log.info("device=cpu   RAM=%.1f GB free   CPU=%d phys", free_ram_gb, cpu_phys)
    log.info("pool_size=%d   threads_per_worker=%d", pool_size, threads_per_worker)

    # Detect and log environment — transparent about what gets enabled.
    env = _detect_env()
    compile_on, compile_mode = _compile_flags(env, "fnn")   # representative arch
    maxtasks = _maxtasks_for_env(env)
    log.info(
        "env=%s   torch_compile=%s  mode=%s   maxtasksperchild=%d",
        env, compile_on, compile_mode, maxtasks,
    )

    t_start = time.time()
    try:
        if pool_size <= 1:
            n_ok, n_skip, n_fail = _run_sequential(trials, log, t_start)
        else:
            n_ok, n_skip, n_fail = _run_parallel_dynamic(
                trials, pool_size, threads_per_worker, device_label, log, t_start,
            )
    except KeyboardInterrupt:
        log.warning("Interrupted by user — partial results preserved.")
        return

    elapsed = time.time() - t_start
    log.info("=" * 72)
    log.info("Grid search complete in %s", _fmt_time(elapsed))
    log.info("  OK=%d  Skipped=%d  Failed=%d  Total=%d", n_ok, n_skip, n_fail, total)
    log.info("Results saved to: %s", DATASET_OUT_ROOT)
    log.info("Analyse results:  PYTHONPATH=. python -m Neural_Networks.analyze_models_grid --models-dir %s", DATASET_OUT_ROOT)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
