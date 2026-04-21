#!/usr/bin/env python3
"""Hyperparameter grid search for BlackBoxFNN, PhysicsRegularizedFNN and
ResidualCorrectionFNN torque models.

ALL configuration is at the top of this file — no CLI arguments.
Edit MODE, ARCH, DRY_RUN, SKIP_EXISTING and the grid dicts, then run::

    PYTHONPATH=. python3 -m Neural_Networks.models.run_loss_residual_grid

Results land in Trained_Models_Grid/ (completely separate from Trained_Models/).
Each architecture sub-directory (FNN/, PhysicsRegularizedFNN/,
ResidualCorrectionFNN/) and a shared models_registry.yaml are created
automatically.

Analyse results afterwards with::

    PYTHONPATH=. python3 -m Neural_Networks.analyze_models_grid

Parallelism strategy
--------------------
* GPU machines  : one trial per GPU, ``n_workers = n_gpus`` concurrent trials.
  Each worker acquires an exclusive GPU slot from a shared queue, sets
  CUDA_VISIBLE_DEVICES, runs the full training loop (AMP + torch.compile),
  then releases the slot.  DataLoader workers are divided fairly across GPUs.
* CPU machines  : single in-process worker (no subprocess overhead; all CPU
  cores available to PyTorch; a memory governor pauses before each trial if
  RAM or swap pressure exceeds configured thresholds).
* Spawn context is used throughout (required for CUDA multi-process safety).

Runtime estimate (HPC mode, 2×A100 80 GB, ~5–10 min/trial with early stopping)
----------------------------------------------------------------------------------
  FNN        :   768 combos  →   ~32–64 h
  PhysicsReg :  2430 combos  →  ~101–202 h
  Residual   :   432 combos  →   ~18–36 h
  Total (all):  3630 combos  →  ~151–302 h  (≈ 6–13 days)
  Run each arch independently with ARCH="fnn" / "physreg" / "residual"
  to spread the load across multiple scheduler allocations.
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

# ============================================================================
# ── CONFIGURATION  (edit here — no CLI arguments) ────────────────────────────
# ============================================================================

# Sweep mode:
#   "local"  → small smoke-test grid (runs in minutes on a laptop / debug node)
#   "hpc"    → exhaustive grid (designed for 2×A100 80 GB)
MODE: str = "local"   # "local" | "hpc"

# Architecture(s) to sweep:
#   "all"      → BlackBoxFNN + PhysicsRegularizedFNN + ResidualCorrectionFNN
#   "fnn"      → BlackBoxFNN only
#   "physreg"  → PhysicsRegularizedFNN only
#   "residual" → ResidualCorrectionFNN only
ARCH: str = "all"   # "all" | "fnn" | "physreg" | "residual"

# Set True to print the full combo table and exit without any training.
DRY_RUN: bool = False

# Set True to skip combos whose output dir already contains a metadata.yaml
# with a matching hyperparameter set (safe resumption after interruption).
SKIP_EXISTING: bool = True

# Dataset run directory (pre-processed CSV produced by preprocess_data.py GUI).
_NN_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_RUN_DIR: str = str(
    _NN_ROOT / "train_data"
    / "run_0419_1338_qraw_d25p3i_ddL_mraw_a1_R_70v15t15_f1p0t1p0_f6a3df"
)

# Output root — completely separate from Trained_Models/ to avoid polluting
# the single-run registry.
MODELS_DIR_ROOT: str = str(_NN_ROOT / "Trained_Models_Grid")
REGISTRY_FILE: str   = str(_NN_ROOT / "Trained_Models_Grid" / "models_registry.yaml")

# ============================================================================
# ── RESOURCE GOVERNOR ────────────────────────────────────────────────────────
#    Prevents system freeze on memory-constrained machines.
#    Trials pause (MEMORY_POLL_INTERVAL seconds) until ALL conditions clear.
# ============================================================================

# PyTorch intra-op thread count per process (0 = auto: n_cpu // n_workers).
CPU_THREADS: int = 0

# Pause when used RAM fraction exceeds this (0–1 scale).
MAX_MEMORY_FRACTION: float = 0.85

# Pause when available RAM falls below this (GB).
MIN_FREE_RAM_GB: float = 2.0

# Pause when swap used exceeds this (GB).
SWAP_THRESHOLD_GB: float = 1.0

# Seconds to sleep between re-checks when throttling.
MEMORY_POLL_INTERVAL: float = 5.0

# ============================================================================
# ── FIXED HYPER-PARAMETERS ───────────────────────────────────────────────────
#    Applied to EVERY trial regardless of which grid combo is being swept.
#    torch_compile / torch_compile_mode are set automatically by the worker.
# ============================================================================
FIXED_HP: dict[str, Any] = {
    "epochs":              5000,     # early stopping cuts this in practice
    "optimizer":           "adamw",
    "lr_scheduler":        "warmup_cosine",
    "early_stopping":      True,
    "early_stop_metric":   "val_rmse",
    "patience":            500,
    "min_delta":           1e-4,
    "grad_clip_norm":      5.0,
    "feature_noise_std":   0.02,
    "data_train_fraction": 1.0,
    "data_train_seed":     0,
    "stride":              1,
    "seed":                42,
    "snapshot_every":      0,
    "print_every":         20,       # log every N epochs (overridden per mode)
}

# Overrides applied on top of FIXED_HP when MODE="local" so smoke-test
# runs finish in minutes rather than hours.
LOCAL_HP_OVERRIDES: dict[str, Any] = {
    "epochs":      10,
    "patience":    10,
    "print_every": 1,   # more frequent so the terminal never feels stuck
}

# ============================================================================
# ── HYPERPARAMETER GRIDS ─────────────────────────────────────────────────────
# ============================================================================

# ── LOCAL  (smoke-test)  ─────────────────────────────────────────────────────
#    FNN: 8 combos   PhysReg: 4 combos   Residual: 4 combos   Total: 16

LOCAL_GRID_FNN: dict[str, list] = {
    # 2 × 2 × 2 × 1 × 1 × 1 = 8 combos
    "hidden_layers": [[128, 256, 128], [256, 512, 256]],
    "dropout":       [0.1, 0.3],
    "learning_rate": [3e-4, 1e-3],
    "weight_decay":  [5e-3],
    "batch_size":    [512],
    "activation":    ["silu"],
}

LOCAL_GRID_PHYSREG: dict[str, list] = {
    # 2 × 1 × 1 × 1 × 1 × 1 × 2 × 1 × 1 = 4 combos
    "hidden_layers":           [[128, 256, 128], [256, 512, 256]],
    "dropout":                 [0.1],
    "learning_rate":           [3e-4],
    "weight_decay":            [5e-3],
    "batch_size":              [512],
    "activation":              ["silu"],
    "physics_weight":          [0.1, 0.5],
    "physics_warmup_fraction": [0.05],
    "phi_lr_ratio":            [0.1],
}

LOCAL_GRID_RESIDUAL: dict[str, list] = {
    # 2 × 1 × 1 × 1 × 1 × 1 × 2 = 4 combos
    "hidden_layers":    [[128, 256, 128], [256, 512, 256]],
    "dropout":          [0.1],
    "learning_rate":    [3e-4],
    "weight_decay":     [5e-3],
    "batch_size":       [512],
    "activation":       ["silu"],
    "alpha_reg_weight": [0.01, 0.05],
}

# ── HPC  (exhaustive — designed for 2×A100 80 GB)  ───────────────────────────

HPC_GRID_FNN: dict[str, list] = {
    # 4 × 4 × 3 × 2 × 4 × 2 = 768 combos
    # With 2×A100 and ~5 min/trial: ~32 h  |  ~10 min/trial: ~64 h
    "hidden_layers": [
        [128, 256, 128],
        [256, 512, 256],
        [512, 1024, 512],
        [256, 512, 512, 256],
    ],
    "dropout":       [0.0, 0.1, 0.2, 0.3],
    "learning_rate": [1e-3, 3e-4, 1e-4],
    "weight_decay":  [5e-3, 1e-2],
    "batch_size":    [512, 1024, 2048, 4096],
    "activation":    ["silu", "gelu"],
}

HPC_GRID_PHYSREG: dict[str, list] = {
    # 3 × 3 × 2 × 1 × 3 × 1 × 5 × 3 × 3 = 810 combos
    # phi_lr_ratio and weight_decay fixed to defaults (lower sensitivity)
    "hidden_layers": [
        [256, 512, 256],
        [512, 1024, 512],
        [256, 512, 512, 256],
    ],
    "dropout":                 [0.0, 0.1, 0.2],
    "learning_rate":           [3e-4, 1e-4],
    "weight_decay":            [5e-3],
    "batch_size":              [512, 1024, 2048],
    "activation":              ["silu"],
    "physics_weight":          [0.05, 0.1, 0.3, 0.5, 1.0],
    "physics_warmup_fraction": [0.02, 0.05, 0.10],
    "phi_lr_ratio":            [0.05, 0.1, 0.2],
}

HPC_GRID_RESIDUAL: dict[str, list] = {
    # 3 × 3 × 2 × 2 × 3 × 1 × 4 = 648 combos
    "hidden_layers": [
        [256, 512, 256],
        [512, 1024, 512],
        [256, 512, 512, 256],
    ],
    "dropout":          [0.0, 0.1, 0.2],
    "learning_rate":    [3e-4, 1e-4],
    "weight_decay":     [5e-3, 1e-2],
    "batch_size":       [512, 1024, 2048],
    "activation":       ["silu"],
    "alpha_reg_weight": [0.0, 0.01, 0.05, 0.1],
}

# ============================================================================
# ── ARCHITECTURE METADATA ────────────────────────────────────────────────────
# ============================================================================

_ARCH_META: dict[str, tuple[str, str, str]] = {
    # key: (model_type, save_subdir, run_help)
    "fnn":      ("BlackBoxFNN",           "FNN",                   "Neural_Networks/models/run_fnn.py"),
    "physreg":  ("PhysicsRegularizedFNN", "PhysicsRegularizedFNN", "Neural_Networks/models/run_physics_regularized.py"),
    "residual": ("ResidualCorrectionFNN", "ResidualCorrectionFNN", "Neural_Networks/models/run_physics_residual.py"),
}

# ============================================================================
# ── INTERNAL HELPERS ─────────────────────────────────────────────────────────
# ============================================================================

# HP keys excluded from the "already-trained?" fingerprint comparison so that
# hardware-dependent settings don't prevent skip detection across machines.
_SKIP_KEYS = frozenset({"torch_compile", "torch_compile_mode", "_grid_seed"})


def _cartesian(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Return all Cartesian-product combos of a grid dict as flat HP dicts."""
    keys = list(grid.keys())
    return [
        {k: v for k, v in zip(keys, combo)}
        for combo in itertools.product(*[grid[k] for k in keys])
    ]


def _build_trials() -> list[dict[str, Any]]:
    """Build the ordered list of trial configs for the current MODE / ARCH."""
    local_grids = {
        "fnn":      LOCAL_GRID_FNN,
        "physreg":  LOCAL_GRID_PHYSREG,
        "residual": LOCAL_GRID_RESIDUAL,
    }
    hpc_grids = {
        "fnn":      HPC_GRID_FNN,
        "physreg":  HPC_GRID_PHYSREG,
        "residual": HPC_GRID_RESIDUAL,
    }
    active_grids = hpc_grids if MODE == "hpc" else local_grids
    active_archs = list(_ARCH_META.keys()) if ARCH == "all" else [ARCH]

    trials: list[dict[str, Any]] = []
    for arch in active_archs:
        model_type, save_subdir, run_help = _ARCH_META[arch]
        for combo in _cartesian(active_grids[arch]):
            hp = {**FIXED_HP, **combo}
            if MODE == "local":
                hp.update(LOCAL_HP_OVERRIDES)
            trials.append({
                "arch":        arch,
                "model_type":  model_type,
                "save_subdir": save_subdir,
                "run_help":    run_help,
                "hp":          hp,
            })
    return trials


def _find_existing_run(subdir_path: Path, model_type: str, hp: dict) -> bool:
    """Return True if *subdir_path* already holds a run matching this HP set."""
    import yaml  # deferred — not available at module import time in workers
    compare_hp = {k: v for k, v in hp.items() if k not in _SKIP_KEYS and not k.startswith("_")}
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


def _hp_short(hp: dict) -> str:
    """One-line HP summary for progress logging."""
    hl = hp.get("hidden_layers", "?")
    hl_str = "×".join(str(x) for x in hl) if isinstance(hl, list) else str(hl)
    return (
        f"hl=[{hl_str}] do={hp.get('dropout','?')} "
        f"lr={hp.get('learning_rate','?'):.0e} wd={hp.get('weight_decay','?'):.0e} "
        f"bs={hp.get('batch_size','?')}"
    )


def _fmt_time(seconds: float) -> str:
    """Format seconds as h:mm:ss."""
    s = int(seconds)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _print_combo_table(trials: list[dict[str, Any]]) -> None:
    """Print a formatted table of all grid combinations."""
    counts = Counter(t["arch"] for t in trials)
    print(f"\n{'='*90}")
    print(f"  GRID SEARCH  —  MODE={MODE!r}  ARCH={ARCH!r}  Total={len(trials)} combos")
    for arch, cnt in sorted(counts.items()):
        print(f"    {arch:<12}  {cnt} combos")
    print(f"{'='*90}")
    print(f"  {'#':<5} {'arch':<10} {'hidden_layers':<25} {'dropout':<8} {'lr':<9} "
          f"{'wd':<8} {'bs':<6} {'extra HPs'}")
    print(f"  {'-'*88}")
    for i, t in enumerate(trials, 1):
        hp = t["hp"]
        hl = hp.get("hidden_layers", [])
        hl_str = "×".join(str(x) for x in hl) if isinstance(hl, list) else str(hl)
        extra_keys = {"physics_weight", "physics_warmup_fraction", "phi_lr_ratio", "alpha_reg_weight"}
        extra = "  ".join(f"{k}={hp[k]}" for k in extra_keys if k in hp)
        print(
            f"  {i:<5} {t['arch']:<10} [{hl_str}]{'':<{max(1,22-len(hl_str))}} "
            f"{hp.get('dropout','?'):<8} {hp.get('learning_rate','?'):<9.0e} "
            f"{hp.get('weight_decay','?'):<8.0e} {hp.get('batch_size','?'):<6} {extra}"
        )
    print(f"{'='*90}\n")


# ============================================================================
# ── WORKER FUNCTION (runs in a spawned subprocess) ───────────────────────────
# ============================================================================

# Module-level GPU queue reference — set once per worker process by the
# ProcessPoolExecutor initializer.  Must be declared at module level so that
# _run_one_trial can access it after spawn re-imports this module.
_GPU_QUEUE: "mp.Queue | None" = None


def _worker_init(gpu_queue: "mp.Queue", n_workers_total: int) -> None:
    """Called once per worker process.  Binds the GPU semaphore queue and
    pins PyTorch intra-op thread count so no worker over-subscribes the CPU."""
    global _GPU_QUEUE
    _GPU_QUEUE = gpu_queue
    import torch as _torch
    _ncpu = os.cpu_count() or 4
    _threads = CPU_THREADS if CPU_THREADS > 0 else max(1, _ncpu // max(1, n_workers_total))
    _torch.set_num_threads(_threads)
    _torch.set_num_interop_threads(max(1, min(4, _threads // 2)))
    if _torch.cuda.is_available():
        _torch.set_float32_matmul_precision("high")   # enable TF32 on Ampere+ GPUs
    # INFO-level so per-epoch progress lines from pipeline.py reach the terminal.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _run_one_trial(trial: dict) -> dict[str, Any]:
    """Train one model configuration.

    Acquires an exclusive GPU slot, trains, and releases the slot.
    Returns a result dict with at minimum: status, arch, hp, gpu_id.
    """
    import os
    import torch
    global _GPU_QUEUE

    # ── Acquire a GPU slot ───────────────────────────────────────────────────
    gpu_id: int = _GPU_QUEUE.get()
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        # CPU mode: hide all GPUs to guarantee CPU execution
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    arch        = trial["arch"]
    model_type  = trial["model_type"]
    save_subdir = trial["save_subdir"]
    run_help    = trial["run_help"]
    hp          = dict(trial["hp"])
    models_dir  = os.path.join(MODELS_DIR_ROOT, save_subdir)
    os.makedirs(models_dir, exist_ok=True)

    try:
        # ── Skip-existing check ──────────────────────────────────────────────
        if SKIP_EXISTING and _find_existing_run(Path(models_dir), model_type, hp):
            return {"status": "skipped", "arch": arch, "hp": hp, "gpu_id": gpu_id}

        # ── Hardware-specific settings ───────────────────────────────────────
        cuda_ok = torch.cuda.is_available() and gpu_id >= 0
        if cuda_ok:
            # max-autotune on HPC for maximum throughput; default on local
            hp["torch_compile"]      = True
            hp["torch_compile_mode"] = "max-autotune" if MODE == "hpc" else "default"
        else:
            hp["torch_compile"]      = False
            hp["torch_compile_mode"] = "default"

        # DataLoader CPU share — GPU mode only.
        # CPU mode: unset NN_NUM_WORKERS entirely so pipeline.py uses its own
        # safe default of 0 DataLoader workers.  Setting it to n_cpu would
        # spawn n_cpu subprocesses each launching a full OMP thread pool,
        # causing catastrophic oversubscription and system freeze.
        n_cpu = os.cpu_count() or 4
        if cuda_ok:
            # Divide available cores fairly across concurrent GPU workers.
            total_gpu_slots = int(os.environ.get("_GRID_N_WORKERS", "1"))
            fair_nw = max(1, n_cpu // max(1, total_gpu_slots))
            os.environ["NN_NUM_WORKERS"] = str(fair_nw)
        else:
            # CPU mode: pipeline.py's default (0 workers) is the correct value.
            os.environ.pop("NN_NUM_WORKERS", None)

        # ── Build and run the training job ───────────────────────────────────
        # Late imports: the spawn context re-runs this module but must not
        # import CUDA-heavy libs at module level before CUDA_VISIBLE_DEVICES is set.
        from Neural_Networks.models.shared.pipeline import TrainJob, run_training
        from Neural_Networks.models.shared.strategies import (
            PLAIN_STRATEGY,
            PHYSICS_REG_STRATEGY,
            RESIDUAL_STRATEGY,
        )

        _strategy_map = {
            "fnn":      PLAIN_STRATEGY,
            "physreg":  PHYSICS_REG_STRATEGY,
            "residual": RESIDUAL_STRATEGY,
        }

        job = TrainJob(
            run_dir=TRAIN_DATA_RUN_DIR,
            models_dir=models_dir,
            registry_file=REGISTRY_FILE,
            model_type=model_type,
            save_subdir=save_subdir,
            hp=hp,
            strategy=_strategy_map[arch],
            run_help=run_help,
        )
        rc = run_training(job)
        return {
            "status": "ok" if rc == 0 else "failed",
            "arch":   arch,
            "hp":     hp,
            "gpu_id": gpu_id,
        }

    except Exception as exc:
        import traceback
        return {
            "status": "error",
            "arch":   arch,
            "hp":     hp,
            "gpu_id": gpu_id,
            "error":  f"{exc}\n{traceback.format_exc()}",
        }
    finally:
        # Always release the GPU slot so other workers can proceed.
        _GPU_QUEUE.put(gpu_id)


# ============================================================================
# ── RESOURCE GOVERNOR HELPERS ────────────────────────────────────────────────
# ============================================================================

def _wait_for_memory(log: logging.Logger) -> None:
    """Block until system memory pressure drops below configured thresholds.

    Three independent conditions must ALL pass before returning:
      1. RAM used fraction ≤ MAX_MEMORY_FRACTION
      2. Available RAM ≥ MIN_FREE_RAM_GB
      3. Swap used ≤ SWAP_THRESHOLD_GB

    Polls every MEMORY_POLL_INTERVAL seconds; emits one warning per wait cycle.
    """
    import psutil

    def _ok() -> bool:
        vm   = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return (
            vm.percent / 100.0 <= MAX_MEMORY_FRACTION
            and vm.available / 1e9 >= MIN_FREE_RAM_GB
            and swap.used / 1e9 <= SWAP_THRESHOLD_GB
        )

    if _ok():
        return

    vm   = psutil.virtual_memory()
    swap = psutil.swap_memory()
    log.warning(
        "Memory pressure: RAM %.0f%% used  %.1f GB free  swap %.1f GB — "
        "pausing %.0f s before next trial",
        vm.percent, vm.available / 1e9, swap.used / 1e9, MEMORY_POLL_INTERVAL,
    )
    while not _ok():
        time.sleep(MEMORY_POLL_INTERVAL)


# ============================================================================
# ── CPU SEQUENTIAL RUNNER ────────────────────────────────────────────────────
# ============================================================================

def _hp_desc(arch: str, hp: dict) -> str:
    """Compact single-line description used as the tqdm bar label."""
    hl = hp.get("hidden_layers", [])
    hl_str = "×".join(str(x) for x in hl) if isinstance(hl, list) else str(hl)
    extra_keys = {
        "physics_weight":  "pw",
        "alpha_reg_weight":"arw",
        "phi_lr_ratio":    "plr",
    }
    extras = "  ".join(
        f"{short}={hp[k]}" for k, short in extra_keys.items() if k in hp
    )
    base = (
        f"{arch:<8} [{hl_str}] "
        f"do={hp.get('dropout','?')} "
        f"lr={hp.get('learning_rate','?'):.0e} "
        f"bs={hp.get('batch_size','?')}"
    )
    return f"{base}  {extras}" if extras else base


def _run_sequential(
    trials: list[dict[str, Any]],
    log: logging.Logger,
    t_start: float,
    *,
    gpu_id: int = -1,
) -> tuple[int, int, int]:
    """Run all trials in-process, one at a time (CPU or single-GPU).

    Each trial shows a live epoch progress bar::

        [ 1/16] fnn [128×256×128] do=0.1 lr=3e-04 bs=512
        [########................]  4/10 ep  val_rmse=0.0821 N·m  pat=1/10

    All per-epoch log.info spam from pipeline.py is suppressed.
    ProcessPoolExecutor is *not* used — no subprocess spawn, no VRAM
    contention, and the memory governor can act between every trial.

    Args:
        gpu_id: GPU index to use (≥0), or -1 for CPU-only.

    Returns (n_ok, n_skip, n_fail).
    """
    total  = len(trials)
    n_ok   = 0
    n_skip = 0
    n_fail = 0

    cuda_ok = gpu_id >= 0 and torch.cuda.is_available()

    # ── Thread and device setup ────────────────────────────────────────────
    _ncpu    = os.cpu_count() or 4
    _threads = CPU_THREADS if CPU_THREADS > 0 else _ncpu
    torch.set_num_threads(_threads)
    torch.set_num_interop_threads(max(1, min(4, _threads // 2)))

    if cuda_ok:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)   # after masking, the target GPU is always 0
        torch.set_float32_matmul_precision("high")   # enable TF32 on Ampere+ GPUs
        # Fair DataLoader workers: use half the CPUs for IO, leave rest for torch
        fair_nw = max(2, min(4, _ncpu // 2))
        os.environ["NN_NUM_WORKERS"] = str(fair_nw)
        _device_label = f"GPU {gpu_id}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.pop("NN_NUM_WORKERS", None)   # pipeline.py safe default: 0
        _device_label = "CPU"

    os.environ["_GRID_N_WORKERS"] = "1"

    # ── Silence pipeline per-epoch INFO scroll ─────────────────────────────
    logging.getLogger("Neural_Networks.models.shared.pipeline").setLevel(logging.WARNING)
    logging.getLogger("Neural_Networks").setLevel(logging.WARNING)

    log.info(
        "Sequential mode — device=%s  threads=%d  %d trials",
        _device_label, _threads, total,
    )

    # Late import after CUDA_VISIBLE_DEVICES is set
    from Neural_Networks.models.shared.pipeline import TrainJob, run_training
    from Neural_Networks.models.shared.strategies import (
        PLAIN_STRATEGY, PHYSICS_REG_STRATEGY, RESIDUAL_STRATEGY,
    )
    _strategy_map = {
        "fnn":      PLAIN_STRATEGY,
        "physreg":  PHYSICS_REG_STRATEGY,
        "residual": RESIDUAL_STRATEGY,
    }

    _w = len(str(total))   # field width for trial counter

    for idx, trial in enumerate(trials, 1):
        arch        = trial["arch"]
        model_type  = trial["model_type"]
        save_subdir = trial["save_subdir"]
        run_help    = trial["run_help"]
        hp          = dict(trial["hp"])
        models_dir  = os.path.join(MODELS_DIR_ROOT, save_subdir)
        os.makedirs(models_dir, exist_ok=True)
        desc = _hp_desc(arch, hp)

        # ── Memory guard ───────────────────────────────────────────────────
        _wait_for_memory(log)

        # ── Skip-existing check ────────────────────────────────────────────
        if SKIP_EXISTING and _find_existing_run(Path(models_dir), model_type, hp):
            n_skip += 1
            tqdm.write(f"  [SKIP]  {idx:{_w}}/{total}  {desc}")
            continue

        # ── Hardware HP overrides ──────────────────────────────────────────
        if cuda_ok:
            hp["torch_compile"]      = True
            hp["torch_compile_mode"] = "max-autotune" if MODE == "hpc" else "default"
        else:
            hp["torch_compile"]      = False
            hp["torch_compile_mode"] = "default"

        # ── Per-trial epoch progress bar ───────────────────────────────────
        epochs_total = int(hp.get("epochs", 5000))
        bar_desc = f"[{idx:{_w}}/{total}] {desc}"
        inner_bar = tqdm(
            total=epochs_total,
            desc=bar_desc,
            unit="ep",
            leave=True,
            dynamic_ncols=True,
            bar_format=(
                "{desc}  "
                "{bar}  "
                "{n_fmt}/{total_fmt} ep"
                "  {postfix}"
            ),
        )
        _last_epoch = [0]

        def _cb(
            epoch: int,
            total_ep: int,
            val_rmse: float,
            pat_ctr: int,
            pat_max: int,
            _bar: tqdm = inner_bar,
            _last: list = _last_epoch,
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
            strategy=_strategy_map[arch],
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
            inner_bar.close()

        elapsed = time.time() - t_start
        eta     = (elapsed / idx) * (total - idx) if idx > 0 else 0.0

        if status == "ok":
            n_ok += 1
            tqdm.write(
                f"  [ OK ]  {idx:{_w}}/{total}  {desc}"
                f"  elapsed={_fmt_time(elapsed)}  ETA={_fmt_time(eta)}"
            )
        elif status == "failed":
            n_fail += 1
            tqdm.write(f"  [FAIL]  {idx:{_w}}/{total}  {desc}  (run_training returned non-zero)")
        else:
            n_fail += 1
            tqdm.write(f"  [ERR ]  {idx:{_w}}/{total}  {desc}\n" + error_msg[:400])

    return n_ok, n_skip, n_fail


# ============================================================================
# ── MAIN ORCHESTRATOR ────────────────────────────────────────────────────────
# ============================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    log = logging.getLogger("grid")

    # ── Build trial list ─────────────────────────────────────────────────────
    trials = _build_trials()
    total  = len(trials)
    arch_counts = Counter(t["arch"] for t in trials)

    # ── Header ──────────────────────────────────────────────────────────────
    log.info("=" * 72)
    log.info("Torque Model  —  Hyperparameter Grid Search")
    log.info("  MODE       : %s", MODE)
    log.info("  ARCH       : %s", ARCH)
    log.info("  TOTAL      : %d trials", total)
    for a, cnt in sorted(arch_counts.items()):
        log.info("    %-10s  %d combos", a, cnt)
    log.info("  DATA       : %s", TRAIN_DATA_RUN_DIR)
    log.info("  OUTPUT     : %s", MODELS_DIR_ROOT)
    log.info("  SKIP_EXIST : %s", SKIP_EXISTING)
    log.info("=" * 72)

    if DRY_RUN:
        log.info("DRY_RUN=True — printing combo table and exiting.")
        _print_combo_table(trials)
        return

    # ── Create output directories ────────────────────────────────────────────
    for arch in _ARCH_META:
        _, save_subdir, _ = _ARCH_META[arch]
        os.makedirs(os.path.join(MODELS_DIR_ROOT, save_subdir), exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR_ROOT, "analysis"), exist_ok=True)

    # ── Detect hardware ──────────────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        n_workers = n_gpus
        log.info("CUDA: %d GPU(s) — running %d concurrent trial(s)", n_gpus, n_workers)
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            log.info("  GPU %d: %s  (%.0f GB VRAM)", i, props.name, props.total_memory / 1024**3)
    else:
        n_workers = 1
        log.info("No CUDA GPUs — running on CPU (sequential, resource-governed)")

    # ── Launch grid search ───────────────────────────────────────────────────
    t_start = time.time()

    if n_workers == 1:
        # ── Single-device path: in-process sequential (CPU or 1 GPU) ─────────
        # No subprocess spawn: no VRAM contention, no torch re-import overhead,
        # memory governor can act between every trial, tqdm bars work cleanly.
        gpu_id = 0 if n_gpus > 0 else -1
        n_ok, n_skip, n_fail = _run_sequential(trials, log, t_start, gpu_id=gpu_id)

    else:
        # ── Multi-GPU path: one spawned process per GPU ───────────────────────
        # Only reached when n_gpus > 1.  Each process owns one GPU exclusively.
        n_ok   = 0
        n_skip = 0
        n_fail = 0

        os.environ["_GRID_N_WORKERS"] = str(n_workers)

        ctx: mp.context.SpawnContext = mp.get_context("spawn")
        gpu_queue: mp.Queue = ctx.Queue()
        for i in range(n_gpus):
            gpu_queue.put(i)

        log.info("Submitting %d trials to ProcessPoolExecutor (max_workers=%d) ...",
                 total, n_workers)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(gpu_queue, n_workers),
        ) as executor:
            future_map = {executor.submit(_run_one_trial, t): t for t in trials}

            for idx, fut in enumerate(as_completed(future_map), 1):
                orig_trial = future_map[fut]
                elapsed = time.time() - t_start
                eta     = (elapsed / idx) * (total - idx) if idx > 0 else 0.0
                try:
                    result = fut.result()
                except Exception as exc:
                    n_fail += 1
                    log.error(
                        "[%d/%d] EXCEPTION  arch=%-9s  %s  err=%s",
                        idx, total, orig_trial["arch"], _hp_short(orig_trial["hp"]), exc,
                    )
                    continue

                status = result.get("status", "?")
                gpu    = result.get("gpu_id", "?")

                if status == "ok":
                    n_ok += 1
                    log.info(
                        "[%d/%d] OK       gpu=%-2s arch=%-9s  %s  elapsed=%s  ETA=%s",
                        idx, total, gpu, result["arch"],
                        _hp_short(result["hp"]),
                        _fmt_time(elapsed), _fmt_time(eta),
                    )
                elif status == "skipped":
                    n_skip += 1
                    log.info(
                        "[%d/%d] SKIPPED  arch=%-9s  %s",
                        idx, total, result["arch"], _hp_short(result["hp"]),
                    )
                else:
                    n_fail += 1
                    log.warning(
                        "[%d/%d] FAILED   gpu=%-2s arch=%-9s  %s  error=%s",
                        idx, total, gpu, result["arch"],
                        _hp_short(result["hp"]),
                        result.get("error", "unknown")[:200],
                    )

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    log.info("=" * 72)
    log.info("Grid search complete in %s", _fmt_time(elapsed_total))
    log.info("  OK=%d  Skipped=%d  Failed=%d  Total=%d", n_ok, n_skip, n_fail, total)
    log.info("Results saved to: %s", MODELS_DIR_ROOT)
    log.info(
        "Analyse results:  PYTHONPATH=. python -m Neural_Networks.analyze_models_grid"
    )
    log.info("=" * 72)


if __name__ == "__main__":
    main()
