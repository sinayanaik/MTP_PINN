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
* GPU machines  : ``multiprocessing.Pool`` with ``maxtasksperchild=1`` so each
  trial runs in a **fresh** process (OS reclaims all memory after every trial).
  Each trial sets ``CUDA_VISIBLE_DEVICES`` from its UI slot *before* importing
  ``torch``.  Several concurrent slots may map to the same GPU when the pool
  is larger than ``n_gpus``; admission control budgets VRAM.  AMP and
  ``torch.compile`` follow your hyperparameters.
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
import sys
import threading
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Do NOT `import torch` at module level: child processes set
# ``CUDA_VISIBLE_DEVICES`` at the start of ``_run_one_trial`` before importing torch.
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
#    probe_resources() probes RAM / VRAM / CPU at startup and derives the
#    optimal n_concurrent, thread counts, and DataLoader workers automatically.
#    Per-trial memory costs are estimated analytically from hidden layer sizes
#    and batch size; RESOURCE_TARGET (70 %) keeps 30 % headroom for the OS.
# ============================================================================

# Fraction of each resource (RAM, VRAM, CPU) the grid search may consume.
# Applied to *currently free* resources, not total — so 0.80 is safe.
RESOURCE_TARGET: float = 0.80

# Hard cap — on large clusters use scheduler array jobs instead.
MAX_CONCURRENT: int = 8

# Memory governor thresholds — pause before a trial if violated.
MIN_FREE_RAM_GB:   float = 2.0
SWAP_THRESHOLD_GB: float = 1.0
MEM_POLL_INTERVAL: float = 5.0

# ============================================================================
# ── RESOURCE PLAN ────────────────────────────────────────────────────────────
# ============================================================================

@dataclass
class TrialMemEst:
    """Analytical memory footprint estimate for one trial."""
    vram_gb:     float   # expected GPU VRAM consumption (incl. CUDA ctx overhead)
    ram_gb:      float   # expected host RAM consumption (incl. torch base RSS)
    param_count: int     # total trainable parameters


@dataclass
class ResourcePlan:
    """Derived execution plan from the system resource probe."""
    n_gpus:               int
    n_concurrent:         int    # parallel trials (hard upper bound for the pool)
    dl_workers_per_trial: int    # DataLoader worker processes per trial
    torch_threads:        int    # PyTorch intra-op threads per trial process
    compile_mode:         str    # torch.compile mode
    use_compile:          bool   # False for local (compile > 10-epoch training)
    use_amp:              bool
    ram_total_gb:         float
    ram_free_gb:          float
    vram_total_gb:        float
    cpu_logical:          int
    cpu_physical:         int
    bottleneck:           str
    cuda_ctx_gb:          float  # measured CUDA context overhead per process
    torch_base_ram_gb:    float  # measured Python+torch RSS per fresh worker


# ── MODEL DIMENSION CONSTANTS (mirror loader.py / torque_models.py) ──────────
_N_INPUTS:  int = 15   # q(5) + qd(5) + qdd(5)
_N_OUTPUTS: int = 5    # τ per joint


def _count_params(
    hidden_layers: list[int],
    in_dim: int = 15,
    out_dim: int = 5,
    extra: int = 0,
) -> int:
    """Count trainable parameters for a build_mlp-style network.

    Architecture (matches torque_models.build_mlp):
        hidden layer i: Linear(in→out) + LayerNorm(out) + Activation + Dropout
        output layer:   Linear(h_last→out_dim)

    ``extra`` covers arch-specific additions, e.g. PhysicsReg tau_scale/bias.
    """
    dims  = [in_dim] + list(hidden_layers) + [out_dim]
    total = 0
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        total += a * b + b      # Linear: weight (a×b) + bias (b)
        if i < len(dims) - 2:   # hidden layers only
            total += 2 * b      # LayerNorm: weight (b) + bias (b)
    return total + extra


def _measure_cuda_ctx_gb(device: int = 0) -> float:
    """Measure the one-time CUDA context overhead for a fresh process.

    Allocates a 1-element tensor to trigger context initialisation, reads
    the VRAM delta via torch.cuda.mem_get_info(), then frees the tensor.
    Returns the overhead in GB.  Falls back to 0.35 GB if CUDA is
    unavailable or the delta looks too small (context already warm).
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0
    try:
        torch.cuda.empty_cache()
        free_before, _ = torch.cuda.mem_get_info(device)
        t = torch.zeros(1, device=f"cuda:{device}")
        free_after, _  = torch.cuda.mem_get_info(device)
        del t
        torch.cuda.empty_cache()
        delta_gb = (free_before - free_after) / 1e9
        return delta_gb if delta_gb > 0.05 else 0.35   # context already warm
    except Exception:
        return 0.35


def _measure_torch_base_ram_gb() -> float:
    """Measure the RSS of a freshly spawned Python+torch worker process.

    Spawns a subprocess that imports psutil and torch, touches a tiny
    tensor, then prints its RSS.  This captures the per-worker base RAM
    overhead (interpreter + PyTorch + CUDA init) that the governor budgets
    for every spawned trial process.  Falls back to 0.5 GB on failure.
    """
    import subprocess
    script = (
        "import psutil, torch; "
        "torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu'); "
        "print(psutil.Process().memory_info().rss)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60,
        )
        rss_bytes = int(result.stdout.strip())
        gb = rss_bytes / 1e9
        return gb if 0.1 < gb < 8.0 else 0.5
    except Exception:
        return 0.5


def _estimate_trial_mem(
    hp: dict,
    arch: str,
    cuda_ctx_gb: float,
    torch_base_ram_gb: float,
) -> TrialMemEst:
    """Analytically estimate GPU VRAM and host RAM for one trial.

    Formula (float32 = 4 bytes throughout — conservative upper bound; real
    AMP activations are fp16 so this never under-estimates):

        P        = trainable parameter count
        model    = P × 4 B
        grads    = P × 4 B
        adam     = P × 2 × 4 B      (exp_avg + exp_avg_sq)
        act      = Σ(h_i) × B × 4 × 2  (fwd + bwd activation cache)
        io_buf   = B × (N_in + N_out) × 4
        raw      = (model + grads + adam + act + io_buf) × 1.5
                    ↑ ×1.5: CUDA allocator fragmentation / cache headroom
        vram_gb  = raw / 1e9 + cuda_ctx_gb         (one context per process)
        ram_gb   = raw / 1e9 + torch_base_ram_gb    (CPU master copy + interp)

    PhysicsRegularizedFNN adds 2×N_OUTPUTS extra params (tau_scale, tau_bias).
    """
    _F  = 4   # bytes per float32
    hl  = hp.get("hidden_layers", [256, 512, 256])
    bs  = int(hp.get("batch_size", 512))
    extra  = 2 * _N_OUTPUTS if arch == "physreg" else 0
    P      = _count_params(hl, in_dim=_N_INPUTS, out_dim=_N_OUTPUTS, extra=extra)

    model_b = P * _F
    grad_b  = P * _F
    adam_b  = P * 2 * _F
    act_b   = sum(hl) * bs * _F * 2
    io_b    = bs * (_N_INPUTS + _N_OUTPUTS) * _F

    raw_b = (model_b + grad_b + adam_b + act_b + io_b) * 1.5
    return TrialMemEst(
        vram_gb     = raw_b / 1e9 + cuda_ctx_gb,
        ram_gb      = raw_b / 1e9 + torch_base_ram_gb,
        param_count = P,
    )


def probe_resources(log: logging.Logger, trials: list[dict]) -> ResourcePlan:
    """Measure system resources and derive the optimal execution plan.

    Per-trial memory costs are computed analytically from hidden layer sizes
    and batch size, calibrated by one-time measurements of the CUDA context
    overhead and the Python+torch worker RSS.  The admission loop in main()
    uses per-trial estimates to decide when to actually submit each trial,
    so n_concurrent here is just the hard upper bound for the worker pool.

    GPU memory sharing
    ------------------
    When n_concurrent > n_gpus, several worker *processes* are pinned to the
    same physical GPU (``CUDA_VISIBLE_DEVICES`` is fixed at worker start).
    The driver multiplexes VRAM — each process is a real separate training
    job; admission control budgets worst-case per-trial VRAM accordingly.
    """
    import psutil
    import statistics
    import torch

    n_trials = len(trials)

    # ── One-time overhead measurements ───────────────────────────────────────
    cuda_ctx_gb       = _measure_cuda_ctx_gb()
    torch_base_ram_gb = _measure_torch_base_ram_gb()
    log.info("  CUDA ctx overhead : %.3f GB  (measured)", cuda_ctx_gb)
    log.info("  Torch worker base : %.3f GB RAM  (measured)", torch_base_ram_gb)

    # ── Per-trial estimates ───────────────────────────────────────────────────
    all_ests  = [
        _estimate_trial_mem(t["hp"], t["arch"], cuda_ctx_gb, torch_base_ram_gb)
        for t in trials
    ]
    vram_vals = sorted(e.vram_gb for e in all_ests)
    ram_vals  = sorted(e.ram_gb  for e in all_ests)
    log.info(
        "  Trial VRAM est  : min=%.3f  med=%.3f  max=%.3f GB",
        vram_vals[0], statistics.median(vram_vals), vram_vals[-1],
    )
    log.info(
        "  Trial RAM  est  : min=%.3f  med=%.3f  max=%.3f GB",
        ram_vals[0],  statistics.median(ram_vals),  ram_vals[-1],
    )

    # ── RAM ──────────────────────────────────────────────────────────────────
    vm           = psutil.virtual_memory()
    ram_total_gb = vm.total / 1e9
    ram_free_gb  = vm.available / 1e9
    n_from_ram   = max(1, int(ram_free_gb * RESOURCE_TARGET / max(ram_vals[-1], 0.1)))

    # ── CPU ──────────────────────────────────────────────────────────────────
    cpu_logical  = os.cpu_count() or 4
    try:
        cpu_physical = psutil.cpu_count(logical=False) or cpu_logical
    except Exception:
        cpu_physical = cpu_logical
    n_from_cpu = max(1, int(cpu_logical * RESOURCE_TARGET) // 2)

    # ── GPU (use *free* VRAM, not total) ─────────────────────────────────────
    n_gpus     = torch.cuda.device_count()
    vram_total = 0.0
    vram_free  = 0.0
    if n_gpus > 0:
        vram_total = sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(n_gpus)
        ) / 1e9
        vram_free = sum(
            torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)
        ) / 1e9
    worst_vram  = vram_vals[-1] if vram_vals else 0.5
    n_from_vram = max(1, int(vram_free * RESOURCE_TARGET / worst_vram)) if n_gpus > 0 else 1

    # ── Concurrency decision ──────────────────────────────────────────────────
    if n_gpus == 0:
        n_concurrent = 1
        bottleneck   = "CPU-only — sequential in-process"
    else:
        n_concurrent = max(1, min(n_from_ram, n_from_vram, n_from_cpu, n_trials, MAX_CONCURRENT))
        if n_from_ram <= n_from_vram and n_from_ram <= n_from_cpu:
            bottleneck = (
                f"RAM ({ram_free_gb:.1f} GB free → {n_from_ram} slots "
                f"at max {ram_vals[-1]:.2f} GB each)"
            )
        elif n_from_vram <= n_from_cpu:
            bottleneck = (
                f"VRAM ({vram_free:.1f} GB free → {n_from_vram} slots "
                f"at max {worst_vram:.2f} GB each)"
            )
        else:
            bottleneck = f"CPU ({cpu_logical} logical → {n_from_cpu} slots at 2 threads min)"

    # ── Per-trial allocations ─────────────────────────────────────────────────
    torch_threads = max(1, int(cpu_logical * RESOURCE_TARGET) // max(1, n_concurrent))
    if n_gpus == 0:
        dl_workers = 0    # CPU mode: workers spawn OMP threads → oversubscription storm
    else:
        dl_workers = max(0, min(4, cpu_logical // max(1, n_concurrent) - torch_threads // 2))

    # ── Compile / AMP ─────────────────────────────────────────────────────────
    # Disable torch.compile for local: compile time (5–30 s) >> 10 training epochs.
    use_compile  = n_gpus > 0 and MODE != "local"
    compile_mode = "max-autotune" if MODE == "hpc" else "default"
    use_amp      = n_gpus > 0

    plan = ResourcePlan(
        n_gpus=n_gpus,
        n_concurrent=n_concurrent,
        dl_workers_per_trial=dl_workers,
        torch_threads=torch_threads,
        compile_mode=compile_mode,
        use_compile=use_compile,
        use_amp=use_amp,
        ram_total_gb=ram_total_gb,
        ram_free_gb=ram_free_gb,
        vram_total_gb=vram_total,
        cpu_logical=cpu_logical,
        cpu_physical=cpu_physical,
        bottleneck=bottleneck,
        cuda_ctx_gb=cuda_ctx_gb,
        torch_base_ram_gb=torch_base_ram_gb,
    )

    log.info("── Resource probe ────────────────────────────────────────────────")
    log.info("  RAM      : %.1f GB total   %.1f GB free", ram_total_gb, ram_free_gb)
    log.info("  CPU      : %d logical   %d physical cores", cpu_logical, cpu_physical)
    if n_gpus > 0:
        for i in range(n_gpus):
            props  = torch.cuda.get_device_properties(i)
            free_i = torch.cuda.mem_get_info(i)[0] / 1e9
            log.info(
                "  GPU %-2d   : %s   %.1f GB total  %.1f GB free",
                i, props.name, props.total_memory / 1e9, free_i,
            )
        log.info(
            "  VRAM     : %.1f GB free  (worst-case trial %.2f GB)",
            vram_free, worst_vram,
        )
    else:
        log.info("  GPU      : none (CPU mode)")
    log.info("  Bottleneck: %s", bottleneck)
    log.info("  → n_concurrent      = %d trial(s) in parallel", n_concurrent)
    log.info("  → torch_threads     = %d per trial", torch_threads)
    log.info("  → dl_workers        = %d per trial", dl_workers)
    log.info("  → use_compile       = %s", use_compile)
    log.info("  → AMP               = %s", use_amp)
    log.info("─────────────────────────────────────────────────────────────────")

    return plan


# ============================================================================
# ── FIXED HYPER-PARAMETERS ───────────────────────────────────────────────────
#    Applied to EVERY trial regardless of which grid combo is being swept.
#    torch_compile / torch_compile_mode are set automatically by the worker.
# ============================================================================
FIXED_HP: dict[str, Any] = {
    "epochs":              5000,     # early stopping cuts this in practice
    "optimizer":           "adamw",
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
    # LR scheduler default parameters (used if not overridden by grid)
    "warm_restart_T_0":    20,       # Cycle length for cosine_warm_restarts
    "warm_restart_T_mult": 1,        # Fixed cycle length (no doubling)
    "warm_restart_eta_min": 3e-6,    # Minimum LR for cosine_warm_restarts
}

# Overrides applied on top of FIXED_HP when MODE="local" so smoke-test
# runs finish in minutes rather than hours.
LOCAL_HP_OVERRIDES: dict[str, Any] = {
    "epochs":      1000,
    "patience":    100,
    "print_every": 1,   # more frequent so the terminal never feels stuck
}

# ============================================================================
# ── HYPERPARAMETER GRIDS ─────────────────────────────────────────────────────
# ============================================================================

# ── LOCAL  (expanded for full local training — no supercomputer needed) ────
#    FNN: 192 combos   PhysReg: 128 combos   Residual: 128 combos   Total: 448

LOCAL_GRID_FNN: dict[str, list] = {
    # 1 × 4 × 1 × 1 × 2 × 1 × 4 × 3 = 96 combos
    "hidden_layers": [[256, 512, 256]],
    "dropout":       [0.0, 0.1, 0.2, 0.3],
    "learning_rate": [3e-4],
    "weight_decay":  [1e-2],
    "batch_size":    [512, 1024],
    "activation":    ["gelu"],
    "data_efficiency_fraction": [1.0, 0.5, 0.25, 0.1],  # 100%, 50%, 25%, 10% of train data
    "lr_scheduler":  ["warmup_cosine", "reduce_on_plateau", "onecycle"],  # 3 scheduler strategies
}

LOCAL_GRID_PHYSREG: dict[str, list] = {
    # 1 × 4 × 1 × 1 × 2 × 1 × 7 × 1 × 1 × 4 × 3 = 336 combos
    "hidden_layers":           [[256, 512, 256]],
    "dropout":                 [0.0, 0.1, 0.2, 0.3],
    "learning_rate":           [3e-4],
    "weight_decay":            [1e-2],
    "batch_size":              [512, 1024],
    "activation":              ["gelu"],
    "physics_weight":          [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],  # expanded
    "physics_warmup_fraction": [0.05],
    "phi_lr_ratio":            [0.1],
    "data_efficiency_fraction": [1.0, 0.5, 0.25, 0.1],  # 100%, 50%, 25%, 10% of train data
    "lr_scheduler":            ["warmup_cosine", "reduce_on_plateau", "onecycle"],  # 3 scheduler strategies
}

LOCAL_GRID_RESIDUAL: dict[str, list] = {
    # 1 × 4 × 1 × 1 × 2 × 1 × 2 × 4 × 3 = 96 combos
    "hidden_layers":    [[256, 512, 256]],
    "dropout":          [0.0, 0.1, 0.2, 0.3],
    "learning_rate":    [3e-4],
    "weight_decay":     [1e-2],
    "batch_size":       [512, 1024],
    "activation":       ["gelu"],
    "alpha_reg_weight": [0.01, 0.05],
    "data_efficiency_fraction": [1.0, 0.5, 0.25, 0.1],  # 100%, 50%, 25%, 10% of train data
    "lr_scheduler":     ["warmup_cosine", "reduce_on_plateau", "onecycle"],  # 3 scheduler strategies
}

# ── HPC  (exhaustive — designed for 2×A100 80 GB)  ───────────────────────────

HPC_GRID_FNN: dict[str, list] = {
    # 4 × 4 × 3 × 2 × 4 × 2 × 4 = 3,072 combos
    # With 2×A100 and ~5 min/trial: ~128 h  |  ~10 min/trial: ~256 h
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
    "lr_scheduler":  ["warmup_cosine", "reduce_on_plateau", "onecycle", "cosine_warm_restarts"],  # 4 scheduler strategies
}

HPC_GRID_PHYSREG: dict[str, list] = {
    # 3 × 3 × 2 × 1 × 3 × 1 × 5 × 3 × 3 × 4 = 3,240 combos
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
    "lr_scheduler":            ["warmup_cosine", "reduce_on_plateau", "onecycle", "cosine_warm_restarts"],  # 4 scheduler strategies
}

HPC_GRID_RESIDUAL: dict[str, list] = {
    # 3 × 3 × 2 × 2 × 3 × 1 × 4 × 4 = 2,592 combos
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
    "lr_scheduler":     ["warmup_cosine", "reduce_on_plateau", "onecycle", "cosine_warm_restarts"],  # 4 scheduler strategies
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
            # Map data_efficiency_fraction → data_train_fraction for pipeline
            if "data_efficiency_fraction" in hp:
                hp["data_train_fraction"] = hp.pop("data_efficiency_fraction")
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
# ── WORKER FUNCTION (spawned; fresh process per trial with Pool maxtasksperchild=1) ─
# ============================================================================
#
# ``multiprocessing.Queue`` must NOT be passed as ``apply_async`` arguments under
# the *spawn* start method (pickle error: "only be shared through inheritance").
# It is injected once per worker via ``Pool(..., initializer=..., initargs=(q,))``.

_POOL_PROGRESS_QUEUE: "mp.Queue | None" = None


def _pool_progress_init(progress_queue: "mp.Queue") -> None:
    """Pool worker initializer: cache the main-process progress queue handle."""
    global _POOL_PROGRESS_QUEUE
    _POOL_PROGRESS_QUEUE = progress_queue


def _configure_grid_child_logging() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    for _ln in (
        "", "Neural_Networks", "Neural_Networks.loader",
        "Neural_Networks.models.shared.pipeline", "Neural_Networks.robot_physics",
    ):
        logging.getLogger(_ln).setLevel(logging.WARNING)


def _run_one_trial(
    trial: dict,
    worker_slot: int,
    plan_dict: dict,
) -> dict[str, Any]:
    """Train one model on GPU ``worker_slot % n_gpus``; env set before ``import torch``.

    ``plan_dict`` is a plain ``dict`` from ``ResourcePlan`` (picklable).  Each
    call may run in a new process (``maxtasksperchild=1``) so the OS reclaims
    all memory when the run finishes.  Progress updates use
    ``_POOL_PROGRESS_QUEUE`` (set in ``_pool_progress_init``).
    """
    import gc
    import os
    import platform

    _pq = _POOL_PROGRESS_QUEUE
    if _pq is None:
        raise RuntimeError("grid worker missing progress queue (pool initializer not run)")

    plan     = ResourcePlan(**plan_dict)
    n_g      = int(plan.n_gpus)
    gpu_id   = int(worker_slot) % n_g if n_g > 0 else -1
    # On Windows, setting CUDA_VISIBLE_DEVICES from a spawned child process is
    # unreliable and can cause DeviceAssert errors; the CUDA driver manages it.
    if platform.system() != "Windows":
        if n_g > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    _configure_grid_child_logging()
    torch.set_num_threads(plan.torch_threads)
    torch.set_num_interop_threads(max(1, min(4, plan.torch_threads // 2)))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

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
            _pq.put(("done", worker_slot, "skip"))
            return {"status": "skipped", "arch": arch, "hp": hp, "gpu_id": gpu_id}

        # ── Hardware-specific settings ───────────────────────────────────────
        cuda_ok = torch.cuda.is_available() and gpu_id >= 0
        hp["torch_compile"]      = plan.use_compile and cuda_ok
        hp["torch_compile_mode"] = plan.compile_mode

        # ``multiprocessing.Pool`` workers are *daemon* processes; they cannot
        # spawn DataLoader worker children ("daemonic processes are not allowed
        # to have children").  Always load batches in-process here.
        os.environ["NN_NUM_WORKERS"] = "0"

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

        # Notify main process: trial starting on this slot.
        _pq.put((
            "start", worker_slot,
            _hp_desc(arch, hp), int(hp.get("epochs", 5000)),
        ))

        def _cb(
            epoch: int, total_ep: int, val_rmse: float,
            pat_ctr: int, pat_max: int,
        ) -> None:
            _pq.put((
                "progress", worker_slot,
                epoch, total_ep, val_rmse, pat_ctr, pat_max,
            ))

        rc = run_training(job, progress_callback=_cb)
        _pq.put(("done", worker_slot, "ok" if rc == 0 else "failed"))
        return {
            "status": "ok" if rc == 0 else "failed",
            "arch":   arch,
            "hp":     hp,
            "gpu_id": gpu_id,
        }

    except Exception as exc:
        import traceback
        if _POOL_PROGRESS_QUEUE is not None:
            _POOL_PROGRESS_QUEUE.put(("done", worker_slot, "error"))
        return {
            "status": "error",
            "arch":   arch,
            "hp":     hp,
            "gpu_id": gpu_id,
            "error":  f"{exc}\n{traceback.format_exc()}",
        }
    finally:
        # Long-lived workers retain CUDA allocator caches; clear between trials
        # so the next run does not stack VRAM and trigger OOM at the system level.
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass


# ============================================================================
# ── RESOURCE GOVERNOR HELPERS ────────────────────────────────────────────────
# ============================================================================

def _wait_for_memory(log: logging.Logger) -> None:
    """Block until RAM and swap pressure drop below configured thresholds.

    Polls every MEM_POLL_INTERVAL seconds; emits one warning per wait cycle.
    """
    import psutil

    def _ok() -> bool:
        vm   = psutil.virtual_memory()
        swap = psutil.swap_memory()
        # On Windows, swap_memory().total is 0 when the pagefile is disabled;
        # treat that as 0 GB used so the check never false-blocks.
        swap_used_gb = swap.used / 1e9 if swap.total > 0 else 0.0
        return (
            vm.available / 1e9 >= MIN_FREE_RAM_GB
            and swap_used_gb <= SWAP_THRESHOLD_GB
        )

    if _ok():
        return

    vm   = psutil.virtual_memory()
    swap = psutil.swap_memory()
    log.warning(
        "Memory pressure: %.1f GB RAM free  %.1f GB swap used — "
        "pausing %.0f s ...",
        vm.available / 1e9, swap.used / 1e9, MEM_POLL_INTERVAL,
    )
    while not _ok():
        time.sleep(MEM_POLL_INTERVAL)


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
    plan: ResourcePlan,
    log: logging.Logger,
    t_start: float,
) -> tuple[int, int, int]:
    """Run all trials in-process, one at a time (CPU or single-GPU).

    Each trial shows a live epoch progress bar::

        [ 1/16] fnn [128×256×128] do=0.1 lr=3e-04 bs=512
        [########................]  4/10 ep  val_rmse=0.0821 N·m  pat=1/10

    All per-epoch log.info spam from pipeline.py is suppressed.
    ProcessPoolExecutor is *not* used — no subprocess spawn, no VRAM
    contention, and the memory governor can act between every trial.

    Returns (n_ok, n_skip, n_fail).
    """
    import torch

    total  = len(trials)
    n_ok   = 0
    n_skip = 0
    n_fail = 0

    import platform
    cuda_ok = plan.n_gpus > 0 and torch.cuda.is_available()

    # ── Thread and device setup ────────────────────────────────────────────
    torch.set_num_threads(plan.torch_threads)
    torch.set_num_interop_threads(max(1, min(4, plan.torch_threads // 2)))

    if cuda_ok:
        if platform.system() != "Windows":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.set_device(0)
        torch.set_float32_matmul_precision("high")
        os.environ["NN_NUM_WORKERS"] = str(plan.dl_workers_per_trial)
        _device_label = "GPU 0"
    else:
        if platform.system() != "Windows":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.pop("NN_NUM_WORKERS", None)
        _device_label = "CPU"

    os.environ["_GRID_N_WORKERS"] = "1"

    # ── Silence pipeline per-epoch INFO scroll ─────────────────────────────
    logging.getLogger("Neural_Networks.models.shared.pipeline").setLevel(logging.WARNING)
    logging.getLogger("Neural_Networks").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    log.info(
        "Sequential mode — device=%s  threads=%d  %d trials",
        _device_label, plan.torch_threads, total,
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
        hp["torch_compile"]      = plan.use_compile and cuda_ok
        hp["torch_compile_mode"] = plan.compile_mode

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
    import torch  # not at module top — workers must not import torch before their initializer sets ``CUDA_VISIBLE_DEVICES``.

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

    # ── Probe resources and build execution plan ─────────────────────────────
    plan    = probe_resources(log, trials)
    t_start = time.time()

    if plan.n_concurrent == 1:
        # ── Sequential path: in-process, tqdm epoch bars per trial ──────────────
        # Used for: CPU-only machines, or GPU with VRAM too tight for 2 trials.
        n_ok, n_skip, n_fail = _run_sequential(trials, plan, log, t_start)

    else:
        # ── Parallel path: dynamic admission control ─────────────────────────
        # Budget is measured ONCE before any worker is spawned, so persistent
        # idle-worker CUDA contexts don't shrink the budget between batches.
        # Concurrency is purely budget-driven — no artificial floor or target
        # count.  MAX_CONCURRENT is the only hard ceiling.
        n_ok   = 0
        n_skip = 0
        n_fail = 0

        import psutil as _psutil

        # ── Static admission budgets (80% of free resources right now) ────────
        if plan.n_gpus > 0:
            vram_budget_gb = torch.cuda.mem_get_info(0)[0] / 1e9 * RESOURCE_TARGET
        else:
            vram_budget_gb = float("inf")
        ram_budget_gb = _psutil.virtual_memory().available / 1e9 * RESOURCE_TARGET

        # Pool size: hard cap is MAX_CONCURRENT; practical cap from resource est.
        pool_size = min(MAX_CONCURRENT, plan.n_concurrent, total)

        log.info(
            "  Admission budget : VRAM=%.2f GB  RAM=%.2f GB  (%.0f%% of free)",
            vram_budget_gb if vram_budget_gb < 1e9 else 0.0,
            ram_budget_gb, RESOURCE_TARGET * 100,
        )
        log.info("  Worker pool size : %d", pool_size)

        plan_dict = asdict(plan)
        os.environ["_GRID_N_WORKERS"] = str(pool_size)

        ctx: mp.context.SpawnContext = mp.get_context("spawn")
        progress_queue: mp.Queue = ctx.Queue()
        # One UI slot 0..pool_size-1 per concurrent trial; recycled when a run ends.
        free_slots:     mp.Queue = ctx.Queue()
        for _i in range(pool_size):
            free_slots.put(_i)

        log.info(
            "Launching Pool — %d processes  maxtasksperchild=1  (%d GPU(s)) ...",
            pool_size, plan.n_gpus,
        )

        # ── Per-slot tqdm bars ────────────────────────────────────────────────
        _w = len(str(total))
        slot_bars = [
            tqdm(
                total=1,
                desc=f"[{i + 1}/{pool_size}] waiting...",
                position=i,
                leave=True,
                dynamic_ncols=True,
                bar_format="{desc}  {bar}  {n_fmt}/{total_fmt} ep  {postfix}",
            )
            for i in range(pool_size)
        ]
        done_bar = tqdm(
            total=total,
            desc="overall",
            position=pool_size,
            leave=True,
            dynamic_ncols=True,
            bar_format="{desc}  {bar}  {n_fmt}/{total_fmt} trials  [{elapsed}<{remaining}, ETA={eta}]  {postfix}",
        )

        # ── Background drain thread ───────────────────────────────────────────
        def _drain() -> None:
            while True:
                msg = progress_queue.get()
                if msg is None:
                    break
                kind = msg[0]
                slot = msg[1]
                bar  = slot_bars[slot]
                if kind == "start":
                    _, slot, desc, total_ep = msg
                    bar.reset(total=total_ep)
                    bar.set_description_str(
                        f"[{slot + 1}/{pool_size}] {desc[:62]}"
                    )
                    bar.set_postfix_str("starting...", refresh=True)
                elif kind == "progress":
                    _, slot, epoch, total_ep, val_rmse, pat_ctr, pat_max = msg
                    delta = epoch - bar.n
                    if delta > 0:
                        bar.update(delta)
                    bar.set_postfix_str(
                        f"rmse={val_rmse:.4f} N·m  pat={pat_ctr}/{pat_max}",
                        refresh=True,
                    )
                elif kind == "done":
                    _, slot, status = msg
                    mark = {"ok": "OK ", "skip": "SKIP", "failed": "FAIL", "error": "ERR"}.get(status, status)
                    bar.set_postfix_str(f"→ {mark}", refresh=True)

        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()

        # ── Per-trial estimates; list sorted largest-VRAM-first ───────────────
        trial_ests = [
            _estimate_trial_mem(
                t["hp"], t["arch"], plan.cuda_ctx_gb, plan.torch_base_ram_gb
            )
            for t in trials
        ]
        # Pending: list of (trial_dict, TrialMemEst), largest first so pop()
        # from the end gives the smallest (used in deadlock guard).
        pending: list = sorted(
            zip(trials, trial_ests),
            key=lambda x: x[1].vram_gb,
            reverse=True,
        )

        # in_flight[0] = estimated VRAM in use, in_flight[1] = estimated RAM.
        # Compared against the STATIC budgets; no real-time re-querying inside
        # the loop so idle worker CUDA contexts don't shrink the budget.
        in_flight = [0.0, 0.0]

        def _try_fill(pool, submitted: dict) -> None:
            """Move trials from *pending* into the pool; budget + free UI slots may cap."""
            import queue as _q
            still: list = []
            idx = 0
            n = len(pending)
            while idx < n:
                trial_t, est_t = pending[idx]
                if (
                    (in_flight[0] + est_t.vram_gb) <= vram_budget_gb
                    and (in_flight[1] + est_t.ram_gb) <= ram_budget_gb
                    and len(submitted) < pool_size
                ):
                    try:
                        _slot = free_slots.get_nowait()
                    except _q.Empty:
                        still.extend(pending[idx:])
                        break
                    _ar = pool.apply_async(_run_one_trial, (trial_t, _slot, plan_dict))
                    submitted[_ar] = (trial_t, est_t, _slot)
                    in_flight[0] += est_t.vram_gb
                    in_flight[1] += est_t.ram_gb
                else:
                    still.append((trial_t, est_t))
                idx += 1
            else:
                pending[:] = still
                return
            pending[:] = still

        def _process_result(
            ares,
            orig_trial: dict,
            est_done: "TrialMemEst",
            free_ui_slot: int,
            done_n: int,
            elapsed: float,
        ) -> None:
            in_flight[0] = max(0.0, in_flight[0] - est_done.vram_gb)
            in_flight[1] = max(0.0, in_flight[1] - est_done.ram_gb)
            free_slots.put(free_ui_slot)
            eta = (elapsed / done_n) * (total - done_n) if done_n > 0 else 0.0
            try:
                result = ares.get()
            except Exception as exc:
                n_fail_ref[0] += 1
                done_bar.update(1)
                done_bar.set_postfix_str(
                    f"ok={n_ok_ref[0]} skip={n_skip_ref[0]} fail={n_fail_ref[0]} ETA={_fmt_time(eta)}",
                    refresh=True,
                )
                tqdm.write(
                    f"  [EXC ]  {orig_trial['arch']}  "
                    f"{_hp_short(orig_trial['hp'])}  err={exc}"
                )
                return

            status = result.get("status", "?")
            gpu    = result.get("gpu_id", "?")
            done_bar.update(1)

            if status == "ok":
                n_ok_ref[0] += 1
                done_bar.set_postfix_str(
                    f"ok={n_ok_ref[0]} skip={n_skip_ref[0]} fail={n_fail_ref[0]} ETA={_fmt_time(eta)}",
                    refresh=True,
                )
                tqdm.write(
                    f"  [ OK ]  gpu={gpu}  {result['arch']}  "
                    f"{_hp_short(result['hp'])}  "
                    f"elapsed={_fmt_time(elapsed)}  ETA={_fmt_time(eta)}"
                )
            elif status == "skipped":
                n_skip_ref[0] += 1
                done_bar.set_postfix_str(
                    f"ok={n_ok_ref[0]} skip={n_skip_ref[0]} fail={n_fail_ref[0]} ETA={_fmt_time(eta)}",
                    refresh=True,
                )
                tqdm.write(
                    f"  [SKIP]  {result['arch']}  {_hp_short(result['hp'])}"
                )
            else:
                n_fail_ref[0] += 1
                done_bar.set_postfix_str(
                    f"ok={n_ok_ref[0]} skip={n_skip_ref[0]} fail={n_fail_ref[0]} ETA={_fmt_time(eta)}",
                    refresh=True,
                )
                tqdm.write(
                    f"  [FAIL]  gpu={gpu}  {result['arch']}  "
                    f"{_hp_short(result['hp'])}  "
                    f"err={result.get('error','?')[:200]}"
                )

        # Use single-element lists so nested closures can mutate them.
        n_ok_ref   = [0]
        n_skip_ref = [0]
        n_fail_ref = [0]
        done_n     = 0

        with ctx.Pool(
            processes=pool_size,
            maxtasksperchild=1,
            initializer=_pool_progress_init,
            initargs=(progress_queue,),
        ) as pool:
            # AsyncResult → (trial_dict, TrialMemEst, ui_slot)
            submitted: dict = {}

            _try_fill(pool, submitted)

            if not submitted and pending:
                trial_t, est_t = pending.pop()   # smallest (end of sorted list)
                log.warning(
                    "Budget too tight — force-submitting smallest trial "
                    "(est. vram=%.2f GB  ram=%.2f GB)",
                    est_t.vram_gb, est_t.ram_gb,
                )
                _fslot = free_slots.get()
                ar = pool.apply_async(_run_one_trial, (trial_t, _fslot, plan_dict))
                submitted[ar] = (trial_t, est_t, _fslot)
                in_flight[0] += est_t.vram_gb
                in_flight[1] += est_t.ram_gb

            while submitted:
                _ready = [a for a in list(submitted.keys()) if a.ready()]
                if not _ready:
                    time.sleep(0.005)
                else:
                    for ar in _ready:
                        o_t, est_d, fslot = submitted.pop(ar)
                        done_n += 1
                        elapsed = time.time() - t_start
                        _process_result(ar, o_t, est_d, fslot, done_n, elapsed)
                _try_fill(pool, submitted)
                if not submitted and pending:
                    trial_t, est_t = pending.pop()   # smallest
                    log.warning(
                        "Budget too tight — force-submitting smallest trial "
                        "(est. vram=%.2f GB  ram=%.2f GB)",
                        est_t.vram_gb, est_t.ram_gb,
                    )
                    _fslot = free_slots.get()
                    ar = pool.apply_async(_run_one_trial, (trial_t, _fslot, plan_dict))
                    submitted[ar] = (trial_t, est_t, _fslot)
                    in_flight[0] += est_t.vram_gb
                    in_flight[1] += est_t.ram_gb

        n_ok   = n_ok_ref[0]
        n_skip = n_skip_ref[0]
        n_fail = n_fail_ref[0]

        # Stop drain thread and close all bars cleanly.
        progress_queue.put(None)
        drain_thread.join(timeout=3)
        for bar in slot_bars:
            bar.close()
        done_bar.close()

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
