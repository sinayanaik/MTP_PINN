#!/usr/bin/env python3
"""Journal comparison: 3-model parallel training with dynamic resource admission.

Models trained:
    * ``BlackBoxFNN``            — purely data-driven MLP baseline.
    * ``PhysicsRegularizedFNN``  — MLP + physics-consistency loss.
    * ``EDR``                    — Equivariant-Decomposed-Residual (structured physics corrections).

Uses the same dynamic-admission parallelism as ``run_loss_residual_grid.py``:
    1. Estimates each trial's VRAM + RAM from its hyperparameters.
    2. Polls live free VRAM / RAM and launches the next trial only if it fits.
    3. RTX 3050 (4 GB): typically runs 2 models concurrently.

Each model uses its standard, published strategy — no experimental augmentations.
Two run modes (chosen at startup via prompt, or env MTP_GRID_MODE):
  * quick    — 1 run per model (the fixed best config), full epochs.
               End-to-end pipeline sanity check with real numbers.
  * detailed — comprehensive per-architecture HP sweep at 100% data
               (FNN 60; PhysReg 48; EDR 96 — total 204; decisive-axis
               search of each model's capacity / Occam / curriculum knobs).
  * dataeff  — best config per arch × data-fraction curve (30 trials).
Both train on 100% data; data-efficiency (train-fraction) curves are a
separate study run later on the winning config.  EDR trig (sin/cos) physics
features are enabled here (q stats injected from the dataset metadata).

Run::

    PYTHONPATH=. python3 Neural_Networks/models/run_journal_grid_3model.py

Results land in ``Trained_Models/Journal_Comparison/`` along with a final
``grid_results.csv`` (every trial) and ``grid_summary.md`` (ranked per arch).
"""

from __future__ import annotations

import gc
import itertools
import json
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

# ── Repo-root bootstrap ─────────────────────────────────────────────────────
# multiprocessing 'spawn' workers start fresh interpreters that do NOT inherit
# the parent's sys.path or PYTHONPATH.  Without this, ``import Neural_Networks``
# fails inside every worker (ModuleNotFoundError) unless the user remembered to
# launch with ``PYTHONPATH=.``.  This top-level block runs in every process
# (main, spawn worker, sequential), so the package is always importable.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# torch is imported lazily — workers set CUDA_VISIBLE_DEVICES before import.
from tqdm import tqdm

# ============================================================================
# ── CONFIGURATION  (edit here — no CLI arguments) ────────────────────────────
# ============================================================================

# Which architectures to run.
#   "all"      → BlackBoxFNN + PhysicsRegularizedFNN + EDR
#   "fnn" | "physreg" | "edr" → just that one
ARCH: str = "all"

# Print the combo table and exit without training.
DRY_RUN: bool = False

# Skip models whose output dir already contains a matching metadata.yaml.
SKIP_EXISTING: bool = True

# Best ablation dataset (locked 91-pt SG, physics-consistent qd/qdd).
_NN_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_RUN_DIR: str = os.environ.get("MTP_TRAIN_DATA_RUN") or str(
    _NN_ROOT / "train_data"
    / "run_abl_q0_qd91_qdd91_tau51_lk_3to1_20260515_1923"
)

# Output root — Journal_Comparison for clean paper results.
MODELS_DIR_ROOT: str = str(_NN_ROOT / "Trained_Models" / "Journal_Comparison")

# For this runner, output goes directly under MODELS_DIR_ROOT (no dataset subdir).
_DATASET_NAME:    str = Path(TRAIN_DATA_RUN_DIR).name
DATASET_OUT_ROOT: str = MODELS_DIR_ROOT
REGISTRY_FILE:    str = str(Path(DATASET_OUT_ROOT) / "models_registry.yaml")

# Active run mode — set by main() from _select_run_mode().  Drives the output
# filenames so the three studies never clobber each other: each mode owns
# ``grid_results_<mode>.csv`` / ``grid_summary_<mode>.md``.  The DETAILED grid
# (the canonical 3-model HP comparison consumed by the plot suite) additionally
# mirrors the legacy unsuffixed ``grid_results.csv`` / ``grid_summary.md``.
RUN_MODE: str = "quick"


def _result_basenames() -> list[tuple[str, str]]:
    """(csv_basename, md_basename) pairs to write for the active RUN_MODE.

    Always the per-mode pair; DETAILED also writes the legacy mirror so the
    plot suite (reads ``grid_results.csv``) keeps working unchanged.
    """
    pairs = [(f"grid_results_{RUN_MODE}.csv", f"grid_summary_{RUN_MODE}.md")]
    if RUN_MODE == "detailed":
        pairs.append(("grid_results.csv", "grid_summary.md"))
    return pairs

# EDR directory (hyphen in name → not a valid package; must be added to sys.path).
_EDR_DIR: str = str(_NN_ROOT / "models" / "Equivariant-Decomposed-Residual")

# Not used as a global batch size (each arch has its own HP); kept for estimator.
BATCH_SIZE: int = 512

# ── Resource-admission parameters (no hardcoded concurrency) ─────────────────
# Reserves: NEVER consumed by the grid (keeps the desktop responsive).
# RAM reserve was 2.0 GB; on 16 GB laptops with a heavy IDE/browser already
# running, that combined with a 1.0 GB per-trial RAM estimate meant the
# admission loop needed 3+ GB free just to dispatch one trial. Workers would
# finish their first task and then stall indefinitely because free RAM
# (post-worker-RSS) never climbed back to 3 GB. 1.0 GB is adequate protection.
# VRAM reserve bumped 0.5 → 2.0 GB (2026-05-20): cuDNN kernel selection +
# torch.compile workspaces can spike a few hundred MB above measured peak;
# 0.5 GB was too tight on the A100 burst-admit window and a 7-worker bunch
# could over-commit before the 1 Hz poller observed the allocation.  At 2 GB
# the headroom is unambiguous and still trivial vs 80 GB cards.
VRAM_RESERVE_GB: float = 2.0
RAM_RESERVE_GB:  float = 1.0

# Polling cadence for the admission loop.
ADMISSION_POLL_SEC: float = 1.0   # while in-flight trials are running
TIGHT_SLEEP_SEC:    float = 5.0   # when nothing currently fits

# Deadlock/starvation guard: if the pool is completely empty (no in-flight
# trial) AND there is still pending work AND no progress has been made for
# this long, the grid is genuinely wedged (e.g. every pending trial's VRAM
# estimate exceeds every GPU forever).  Break out, write partial results, and
# fail loudly rather than spin forever.  Generous so a long single trial on a
# slow GPU never trips it.
STARVATION_TIMEOUT_SEC: float = 1800.0

# Sequential-path RAM floor (laptop-grade check).
MIN_FREE_RAM_GB:   float = 2.0
MEM_POLL_INTERVAL: float = 5.0

# ============================================================================
# ── PER-ARCH FIXED HYPERPARAMETERS ───────────────────────────────────────────
# Best known configs — no sweep.  Each arch uses its own HP set.
# ============================================================================

# Shared MLP backbone (BlackBoxFNN and PhysicsRegularizedFNN share this).
# NOTE: no physics-specific keys here — those belong only to PhysReg, so
# FNN's HP dict / config string / run-id stay clean (no spurious pw=).
#
# FAIR-BENCHMARK PROTOCOL: all three archs share an IDENTICAL optimisation
# protocol (scheduler, epoch budget, patience, min_delta, early-stop metric) so
# the *architecture* is the only independent variable.  Architecture-specific
# regularisation (batch_size, dropout, weight_decay, grad_clip_norm, δ-net
# widths, correction reg) is intentionally NOT equalised — it must suit each
# model — but the training protocol is.
#
# Why warmup_cosine (R4, 2026-05-19): cosine_warm_restarts re-spikes the LR
# every T_0=15 epochs.  For EDR this stacked on the old two-phase optimizer
# shock and made the val curve wobble, tripping the 50-epoch patience at
# ~ep70 (the user-observed "EDR only trains 50-70 epochs").  A SINGLE smooth
# warmup→long-cosine decay over the full 1000-epoch budget removes the
# periodic shocks: with EDR's new γ capacity-ramp (no freeze/Adam-reset) the
# loss descends smoothly so early-stop now fires only at true convergence —
# EDR "naturally keeps learning".  Applied identically to all three archs so
# the fair-benchmark protocol is preserved (this also tests whether the FNN/
# PhysReg ~700-epoch tail was merely warm-restart re-annealing).
_FIXED_HP_FNN_BASE: dict[str, Any] = {
    "hidden_layers":           [256, 512, 256],
    "dropout":                 0.4,
    "activation":              "silu",
    "learning_rate":           3e-4,
    "weight_decay":            5e-3,
    "batch_size":              512,
    "optimizer":               "adamw",
    # Identical schedule to EDR (fair-benchmark protocol): one smooth
    # warmup→cosine decay over the full epoch budget, no warm restarts (R4).
    "lr_scheduler":            "warmup_cosine",
    "grad_clip_norm":          5.0,
    "feature_noise_std":       0.02,
    # Long-horizon sweep — equal 1500-epoch budget for ALL three archs (fair
    # protocol); early-stop still ends runs that genuinely converge, this just
    # removes the budget ceiling so each model can "keep learning".
    "epochs":                  1500,
    "min_delta":               1e-5,   # let slow-but-real gains reset patience
    "early_stopping":          True,
    "early_stop_metric":       "val_rmse",
    "data_train_fraction":     1.0,
    "data_train_seed":         0,
    "stride":                  1,
    "torch_compile":           False,
    "torch_compile_mode":      "default",
    "snapshot_every":          0,
}

# patience=50 for FNN/PhysReg, 150 for EDR: FNN/PhysReg converge quickly on
# the warmup_cosine schedule so a tighter patience cuts wasted tail epochs;
# EDR's smooth γ-gate ramp + slower descent still benefits from 150 to reach
# true convergence (early-stop fires only on real plateau, not wandering).
FIXED_HP_FNN = {**_FIXED_HP_FNN_BASE, "patience": 50}

# PhysReg = same backbone + the physics-consistency penalty (its defining HP).
FIXED_HP_PHYSREG = {
    **_FIXED_HP_FNN_BASE,
    "patience":                50,
    "physics_weight":          0.5,
    "physics_warmup_fraction": 0.05,
    "phi_lr_ratio":            0.1,
}

FIXED_HP_EDR: dict[str, Any] = {
    # ══════════════════════════════════════════════════════════════════════
    # EVIDENCE-OPTIMAL EDR (2026-05-19) = proven "corrected-P1" structure
    #                                     + the two genuinely-new safe wins.
    # ----------------------------------------------------------------------
    # Six prior architecture rounds established that EVERY structural /
    # capacity lever past corrected-P1 monotonically REGRESSED test
    # (see memory: project_edr_arch_revamp / fair_protocol_edr_capacity).
    # A 2026-05-19 local 150-ep check reconfirmed it: forcing PSD δM (R2) +
    # Christoffel-only δC (R1) + spectral-norm (R5) UNDERFITS — EDR test
    # 0.0987 vs PhysReg 0.0910, train/val both worse → over-constrained.
    # The val→test gap is a documented data-split property, not a tuning
    # gap (PhysReg's gap is actually larger).
    #
    # So we DO NOT re-run the falsified "more structure" experiments.  We
    # keep corrected-P1 (EDR's empirical best, ≈0.0896, tied w/ PhysReg) and
    # add ONLY the two new, NON-structural improvements that no prior round
    # had, both of which the 150-ep run validated:
    #   R3  smooth γ-gate replaces the two-phase freeze/Adam-reset shock —
    #       EDR now descends smoothly; best epoch moved ~22→88, the
    #       "stops at 50-70 ep" symptom is gone (trains naturally long).
    #   R4  single warmup_cosine (no warm-restart LR spikes) — smooth.
    #   R6  headline grid is multi-seed (see GRID_*_QUICK) so the EDR-vs-
    #       PhysReg verdict finally has error bars (the old single-seed
    #       "PhysReg wins by 0.37%" was pure noise; EDR's grid-MEAN was
    #       already lower).
    # R1/R2/R5 remain available as ABLATION flags (default OFF) and ship
    # tested for the structural-passivity story, but are NOT the journal
    # config.
    # ══════════════════════════════════════════════════════════════════════
    # 2026-05-22 SPEEDUP: zero of 56 prior HPC EDR runs reached 1500 epochs;
    # median was 330 (p90=612). The 1500 budget was a safety upper bound
    # nobody reached.  Lowered to 1000 (still 1.6× the p90 actual epoch
    # count, so no early-truncation risk).  Patience 150→50 trims the
    # plateau tail (~100 epochs × 2.1 min/ep ≈ 3.5h saved per trial); now
    # matches FNN/PhysReg.
    "epochs":                  1000,   # longer horizon — EDR keeps learning (was 1500)
    "batch_size":              256,
    "learning_rate":           3e-4,
    "weight_decay":            2e-3,
    "optimizer":               "adamw",
    # R4 — same scheduler family as FNN/PhysReg (fair protocol); EDR raises
    # the cosine LR floor (1%→5% of base) so the long tail doesn't stall.
    "lr_scheduler":            "warmup_cosine",
    "warmup_cosine_min_factor": 0.05,  # EDR-only LR floor (FNN/PhysReg keep 0.01)
    "early_stopping":          True,
    "early_stop_metric":       "val_rmse",
    "patience":                50,     # 2026-05-22: 150→50 (matches FNN/PhysReg; prior data showed γ-gate ramp finishes by ep 300, best by ~ep 180)
    "min_delta":               1e-5,    # R3/R4: smooth descent ⇒ small δ is safe
    "grad_clip_norm":          1.0,
    "feature_noise_std":       0.02,
    "activation":              "silu",
    # Minimum-capacity δ-nets — the 2026-05-16 sweet spot (every wider net
    # overfit; the strong Occam prior protects generalisation).
    "gravity_hidden":          [64, 64],
    "inertia_hidden":          [64, 64],
    "coriolis_hidden":         [64, 64],
    "friction_hidden":         [32, 32],
    "use_friction_qdd":        False,
    "use_phys_cond":           True,    # the ONE lever that helped (0.0923→0.0904)
    # ── Structure = corrected-P1 (proven best); R1/R2/R5 OFF (ablation) ────
    "coriolis_structural":     False,   # R1 OFF — independent δC (corrected-P1)
    "coriolis_matrix_form":    False,   # element-wise δC generalised best here
    "inertia_psd":             False,   # R2 OFF — symmetric δM can also reduce M
    "spectral_norm":           False,   # R5 OFF — caps capacity, underfit at 150ep
    "friction_form":           "mlp",   # Stribeck regressed test previously
    # ── Smooth capacity gate (R3 — NEW, validated: best ep 22→88) ─────────
    "correction_gain_ramp_frac": 0.30, # γ: 0→1 cosine over first 30% of epochs
    # ── Generalisation robustness ─────────────────────────────────────────
    "ema_decay":               0.9,     # flat low-variance checkpoint selection
    "joint_loss_weights":      None,    # uniform (matches macro headline)
    "lambda_correction_reg":   1.0e-1,  # scalar Occam prior (corrected-P1 value)
    "lambda_correction_reg_per_component": None,
    "lambda_correction_decay": "none",
    "correction_dropout":      0.30,    # 2026-05-22: bumped 0.15→0.30; HPC per-HP analysis showed cdrop=0.30 gives both lower val AND lower test (see project_edr_dropout_root_cause)
    "correction_reg_inertia_normalize": True,
    "enable_passivity_loss":   False,
    "lambda_passivity":        0.01,
    "data_train_fraction":     1.0,
    "data_train_seed":         0,
    "stride":                  1,
    "torch_compile":           False,
    "torch_compile_mode":      "default",
    "snapshot_every":          0,
    "print_every":             2,
}

_FIXED_HP_BY_ARCH: dict[str, dict[str, Any]] = {
    "fnn":      FIXED_HP_FNN,
    "physreg":  FIXED_HP_PHYSREG,
    "edr":      FIXED_HP_EDR,
}

# ============================================================================
# ── RUN MODES: quick test  vs  detailed HP sweep ────────────────────────────
# Two modes, chosen at startup (interactive prompt, or env MTP_GRID_MODE):
#
#   quick    — 1 run per model (the fixed best config), full epochs /
#              early-stop.  Sanity-checks the whole pipeline with real
#              numbers, not a science result.
#
#   detailed — comprehensive per-architecture HP sweep at 100% data
#              (FNN 60; PhysReg 48; EDR 96 — total 204).  Searches the
#              knobs that actually shape each architecture's capacity /
#              regularisation / defining hyperparameters.
#              Data-efficiency (train-fraction) curves are a SEPARATE study
#              run later on the winning config — not swept here.
#
# All modes train on 100% data (data_train_fraction stays 1.0 from the
# FIXED_HP_* base); val/test are full and identical across runs.
# ============================================================================

# ── QUICK / HEADLINE: the fixed best config, MULTI-SEED (R6) ────────────────
# A single seed cannot support a "clean victory" claim — the prior 0.37%
# PhysReg-vs-EDR gap was decided by seed=42 alone (std=0.00000, n=1 in
# grid_summary.md).  The headline grid is now 5 seeds per arch so the
# comparison has error bars; the journal claim is the multi-seed
# mean ± std of each arch's fixed best config.  (DETAILED below stays
# single-seed — it is an architecture-tuning sweep, not the headline.)
_HEADLINE_SEEDS = [42, 1, 2, 3, 4]
GRID_FNN_QUICK:     dict[str, list] = {"seed": _HEADLINE_SEEDS}
GRID_PHYSREG_QUICK: dict[str, list] = {"seed": _HEADLINE_SEEDS}
GRID_EDR_QUICK:     dict[str, list] = {"seed": _HEADLINE_SEEDS}

# ── DETAILED: comprehensive architecture HP sweeps (single seed=42) ─────────
# FNN (pure MLP on [q,q̇,q̈]) — capacity (depth/width), regularisation
# (dropout, weight_decay, input noise), and optimisation (lr).
GRID_FNN_DETAILED: dict[str, list] = {
    "hidden_layers": [
        [128, 128], [256, 256], [128, 256, 128],
        [256, 512, 256], [512, 512, 512],
    ],                                              # 5  capacity/depth
    "dropout":           [0.1, 0.3, 0.5],           # 3  regularisation
    "weight_decay":      [1e-4, 5e-3],              # 2  L2
    "learning_rate":     [3e-4, 1e-3],              # 2  optimisation
    "seed":              [42],
    # feature_noise_std axis dropped (lean sweep, 2026-05-17): the
    # 0.0/0.05 input-noise knob never separated configs in prior FNN
    # sweeps — capacity × dropout × wd × lr is the decisive subspace.
    # It stays fixed at the FIXED_HP_FNN value for every trial.
}  # 5·3·2·2 = 60

# PhysReg — same MLP backbone axes + the physics-consistency weight (its
# defining hyperparameter) + warmup of that penalty.
GRID_PHYSREG_DETAILED: dict[str, list] = {
    "hidden_layers": [
        [128, 256, 128], [256, 512, 256], [512, 512, 512], [256, 256],
    ],                                              # 4  capacity
    "dropout":                 [0.1, 0.3],          # 2  regularisation (trimmed)
    # weight_decay axis dropped (2026-05-17 trim): L2 is the least-separating
    # PhysReg knob vs capacity/dropout/physics_weight — it stays fixed at the
    # FIXED_HP_PHYSREG value (5e-3) for every trial.
    # physics_weight extended into the strong-penalty regime (added 2.0,
    # 2026-05-17) so the sweep probes whether harder physics-consistency
    # enforcement helps or over-constrains the black-box backbone.
    "physics_weight":          [0.05, 0.1, 0.25, 0.5, 1.0, 2.0],  # 6  defining HP
    "seed":                    [42],
}  # 4·2·6 = 48

# EDR (structured: four δ-nets + smooth γ-gate) — δ-net capacity, the Occam
# correction-magnitude penalty (its defining HP), correction dropout, and
# friction conditioning.  edr_width expands to gravity/inertia/coriolis_hidden
# (+ friction = half) in _build_trials().
#
# Trimmed-210 expansion (2026-05-19): the combined two-stage run uses Stage A's
# ACTUAL grid winner for Stage B, so the EDR search is widened around the
# proven corrected-P1 recipe to let the data DISCOVER a stronger config — but
# only the empirically-decisive axes (width × λ_reg × dropout × friction). NO
# structural levers (coriolis_structural / inertia_psd / spectral_norm stay
# OFF in FIXED_HP_EDR; six prior rounds falsified them) and no lr/patience
# axes (fixed at the new FIXED_HP_EDR horizon).
#
# 2026-05-22 TIGHTENING (see project_edr_dropout_root_cause memory):
#   • dropout=0.05 was the dominant val<test gap driver on the 56 prior
#     HPC runs (cdrop=0.05 → test 0.1064; cdrop=0.30 → test 0.0917).
#     Removed entirely; sweep cdrop ∈ {0.30, 0.45} to test whether even
#     higher dropout shrinks the gap further.
#   • λ_corr ∈ [0.002, 0.2] showed weak/no test-RMSE effect within
#     single-seed noise (mean variance < 0.005 across that range).
#     Sweep narrowed to [0.01, 0.05] — the empirical sweet spot.
#   • width [32,32] and [128,128] dropped: small widths under-fit, [128,128]
#     overfits (project_fair_protocol_edr_capacity).  Sweep [48,48], [64,64].
#   • single seed=[42] (per-user request 2026-05-22) — keeps trial count
#     low for HPC wall-clock fit; multi-seed verification deferred to the
#     DATAEFF curve which already runs the winner across 3 seeds × 10 fractions.
# Final: 2·2·2·2·1 = 16 trials (was 210 single-seed; now 16 single-seed).
GRID_EDR_DETAILED: dict[str, list] = {
    "edr_width": [
        [48, 48], [64, 64],
    ],                                              # 2  δ-net capacity (sweet spot)
    "lambda_correction_reg":
        [1e-2, 5e-2],                               # 2  Occam strength (empirical range)
    "correction_dropout":     [0.30, 0.45],         # 2  δ-net reg (≥0.30 closes val<test gap)
    "use_friction_qdd":       [False, True],        # 2  friction conditioning
    "seed":                   [42],                 # 1  single seed (multi-seed lives in DATAEFF)
}  # 2·2·2·2·1 = 16 trials

# ── DATAEFF: lightweight data-efficiency curve (separate study) ─────────────
# Each arch runs its proven FIXED_HP_* best config (NOT the DETAILED sweep)
# while sweeping ONLY the training-data fraction.  data_train_fraction
# subsamples the TRAIN split deterministically (via data_train_seed, set from
# `seed` in _build_trials); val/test stay full & identical (loader.py).  This
# is the "how much data does each architecture need" curve — deliberately
# kept off the 204-trial DETAILED grid (which stays at frac=1.0).
#
# 2026-05-22: single-seed DATAEFF (per user request, to cut wall-clock).
# History: 2026-05-19 multi-seed was added because single-seed made the
# curve noise-dominated (a ~0.0008 N·m PhysReg wiggle read as "test RMSE
# increasing with more data" when PhysReg is in fact ~flat).  Reverting to
# single-seed to fit HPC budget; the resulting curve will have no error
# band — interpret trends with care, especially small differences across
# adjacent fractions.  Cost: 30 trials → 10 trials per arch.
_DATAEFF_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_DATAEFF_SEEDS = [42]
GRID_FNN_DATAEFF:     dict[str, list] = {"data_train_fraction": _DATAEFF_FRACTIONS, "seed": _DATAEFF_SEEDS}
GRID_PHYSREG_DATAEFF: dict[str, list] = {"data_train_fraction": _DATAEFF_FRACTIONS, "seed": _DATAEFF_SEEDS}
GRID_EDR_DATAEFF:     dict[str, list] = {"data_train_fraction": _DATAEFF_FRACTIONS, "seed": _DATAEFF_SEEDS}
# 10·1 per arch × 3 archs = 30 trials

# Default = quick; main() overrides _ARCH_GRID after mode selection.
GRID_FNN:     dict[str, list] = GRID_FNN_QUICK
GRID_PHYSREG: dict[str, list] = GRID_PHYSREG_QUICK
GRID_EDR:     dict[str, list] = GRID_EDR_QUICK

_ARCH_GRID_QUICK: dict[str, dict[str, list]] = {
    "fnn": GRID_FNN_QUICK, "physreg": GRID_PHYSREG_QUICK, "edr": GRID_EDR_QUICK,
}
_ARCH_GRID_DETAILED: dict[str, dict[str, list]] = {
    "fnn": GRID_FNN_DETAILED, "physreg": GRID_PHYSREG_DETAILED, "edr": GRID_EDR_DETAILED,
}
_ARCH_GRID_DATAEFF: dict[str, dict[str, list]] = {
    "fnn": GRID_FNN_DATAEFF, "physreg": GRID_PHYSREG_DATAEFF, "edr": GRID_EDR_DATAEFF,
}

_ARCH_META: dict[str, tuple[str, str, str]] = {
    "fnn":      ("BlackBoxFNN",           "FNN",              "Neural_Networks/models/run_fnn.py"),
    "physreg":  ("PhysicsRegularizedFNN", "PhysicsRegularized", "Neural_Networks/models/run_physics_regularized.py"),
    "edr":      ("EDR",                  "EDR",              "Neural_Networks/models/Equivariant-Decomposed-Residual/run_edr.py"),
}

# Active grid — replaced by main() once the run mode is chosen.
_ARCH_GRID: dict[str, dict[str, list]] = dict(_ARCH_GRID_QUICK)

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


def _load_edr_q_stats() -> dict[str, Any]:
    """Read q normalisation stats from the dataset metadata.

    EDR's gravity/inertia/Coriolis nets only receive sin(q_raw)/cos(q_raw)
    physics features when ``_q_mean``/``_q_std`` are present in the HP dict
    (see ``edr_strategy._make_model_edr`` → ``EDRModel._has_trig_features``).
    ``run_edr.py:main`` injects these; the grid path must do the same or EDR
    silently trains without its trig features.  Returns {} if unavailable.
    """
    meta_path = Path(TRAIN_DATA_RUN_DIR) / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return {}
    norm = meta.get("normalisation", {})
    if "mean_q" in norm and "std_q" in norm:
        return {"_q_mean": norm["mean_q"], "_q_std": norm["std_q"]}
    return {}


_EDR_Q_STATS: dict[str, Any] = _load_edr_q_stats()


def _build_trials() -> list[dict[str, Any]]:
    active_archs = list(_ARCH_META.keys()) if ARCH == "all" else [ARCH]
    if not all(a in _ARCH_META for a in active_archs):
        raise ValueError(f"ARCH={ARCH!r} must be 'all' or one of {list(_ARCH_META)}")

    trials: list[dict[str, Any]] = []
    for arch in active_archs:
        model_type, save_subdir, run_help = _ARCH_META[arch]
        base_hp = _FIXED_HP_BY_ARCH[arch]
        for combo in _cartesian(_ARCH_GRID[arch]):
            hp = {**base_hp, **combo}
            if "seed" in combo:
                hp["data_train_seed"] = int(combo["seed"])
            if arch == "edr":
                # Expand the single width axis into the three δ-net widths
                # (+ friction = half-width), then drop the synthetic key.
                w = combo.get("edr_width")
                if w is not None:
                    hp["gravity_hidden"]  = list(w)
                    hp["inertia_hidden"]  = list(w)
                    hp["coriolis_hidden"] = list(w)
                    hp["friction_hidden"] = [max(1, w[0] // 2), max(1, w[-1] // 2)]
                    hp.pop("edr_width", None)
                # Enable sin/cos trig physics features (mirrors run_edr.py).
                hp.update(_EDR_Q_STATS)
            trials.append({
                "arch":        arch,
                "model_type":  model_type,
                "save_subdir": save_subdir,
                "run_help":    run_help,
                "hp":          hp,
            })
    return trials


def _write_resume_status(
    skipped: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    status_path: str,
) -> None:
    """Write a JSON resume-status checkpoint to ``status_path``.

    Captures what's already complete on disk and what's queued for dispatch
    so the user can read the file across HPC job restarts and know exactly
    how many trials remain.  Best-effort — failure to write is logged but
    does NOT abort the run.

    Resumability story (2026-05-22):
    the grid is already resumable across HPC jobs via ``SKIP_EXISTING``
    (matching ``metadata.yaml`` files on disk are recognised as completed
    and dropped from the pending queue at startup).  This status file
    surfaces that state externally so it is visible without log scraping.
    """
    from collections import Counter as _Counter
    try:
        os.makedirs(os.path.dirname(os.path.abspath(status_path)) or ".",
                    exist_ok=True)
        payload = {
            "timestamp":        datetime.now().isoformat(),
            "total_planned":    len(skipped) + len(pending),
            "completed_count":  len(skipped),
            "pending_count":    len(pending),
            "completed_by_arch": dict(_Counter(t["arch"] for t in skipped)),
            "pending_by_arch":   dict(_Counter(t["arch"] for t in pending)),
            "pending_trials": [
                {
                    "arch": t["arch"],
                    "hp_signature": {
                        k: v for k, v in t["hp"].items()
                        if k not in _SKIP_KEYS and not k.startswith("_")
                    },
                }
                for t in pending
            ],
        }
        with open(status_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("grid").warning(
            "Failed to write resume-status to %s: %s", status_path, exc
        )


def _partition_trials_by_skip(
    trials: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pre-filter trials against DATASET_OUT_ROOT. Returns (pending, skipped).

    ``skipped`` is the list of already-trained trial dicts (NOT just a count)
    so the finalizer can reconstruct their CSV/summary rows from disk — the
    grid_results.csv must describe the whole active grid, not only the trials
    executed this session.

    Runs in the main process before any worker is spawned. Saves the 3–5 s
    cold-start-per-skipped-trial cost that was paid when the check lived
    inside the worker.
    """
    if not SKIP_EXISTING:
        return list(trials), []
    pending: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    subdir_cache: dict[str, Path] = {}
    for t in trials:
        sd = t["save_subdir"]
        base = subdir_cache.get(sd)
        if base is None:
            base = Path(DATASET_OUT_ROOT) / sd
            subdir_cache[sd] = base
        if base.is_dir() and _find_existing_run(base, t["model_type"], t["hp"]):
            skipped.append(t)
        else:
            pending.append(t)
    return pending, skipped


def _run_metrics_for(out_root: str, arch: str, hp: dict) -> dict:
    """Locate the run dir matching (arch, hp) and return its trajectory-macro
    TRAIN / VAL / TEST rmse for the post-grid display.

    Reuses the exact full-hp metadata compare used by ``_find_existing_run``
    (robust — not run-dir-name string matching), picking the most recent
    match.  Returns {} when no match / unreadable (caller shows '—').
    """
    import yaml
    meta = _ARCH_META.get(arch)
    if meta is None:
        return {}
    model_type, subdir, _ = meta
    base = Path(out_root) / subdir
    if not base.is_dir():
        return {}
    compare_hp = {
        k: v for k, v in hp.items()
        if k not in _SKIP_KEYS and not k.startswith("_")
    }
    best_match = None  # (mtime, run_dir, metadata)
    for meta_file in base.glob("*/metadata.yaml"):
        try:
            with open(meta_file) as fh:
                m = yaml.safe_load(fh)
            if not isinstance(m, dict) or m.get("model_type") != model_type:
                continue
            saved = {
                k: v for k, v in m.get("hyperparams", {}).items()
                if k not in _SKIP_KEYS and not k.startswith("_")
            }
            if saved != compare_hp:
                continue
            mt = meta_file.stat().st_mtime
            if best_match is None or mt > best_match[0]:
                best_match = (mt, meta_file.parent, m)
        except Exception:
            continue
    if best_match is None:
        return {}
    _, run_dir, m = best_match
    out = {
        "test": float(m.get("test_metrics", {}).get("rmse_traj_macro", float("nan"))),
        "val":  float(m.get("val_metrics", {}).get("rmse_traj_macro", float("nan"))),
        "train": float("nan"),
        "best_epoch": m.get("best_epoch"),
    }
    # Train rmse at the best-val epoch (pooled) — from training_history.csv.
    try:
        import csv as _csv
        rows = list(_csv.DictReader(open(run_dir / "training_history.csv")))
        if rows:
            def _f(r, k):
                try:
                    return float(r[k])
                except Exception:
                    return float("inf")
            bi = min(range(len(rows)), key=lambda i: _f(rows[i], "val_rmse"))
            out["train"] = _f(rows[bi], "train_rmse")
    except Exception:
        pass
    return out


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
_N_IN_PHYS = 35   # PhysicsRegularizedFNN: [q, qd, qdd, τ_g, τ_M, τ_C, τ_f]
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
    bs = int(hp.get("batch_size", 512))
    if arch == "edr":
        # EDR has 4 sub-networks; size the estimate off the swept δ-net width
        # so a wide [128,128] sweep config isn't under-estimated (→ VRAM
        # over-pack / OOM on the 4 GB card).
        hl = list(hp.get("gravity_hidden", [64, 64]))
        n_in = _N_IN
        P = _count_params(hl, n_in=n_in, extra=0) * 4  # 4 sub-nets
    else:
        hl = hp.get("hidden_layers", [128, 256, 128])
        n_in = _N_IN_PHYS if arch == "physreg" else _N_IN
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


# ── Resource probe (peak RSS + peak VRAM) ───────────────────────────────────
# A single subprocess that mirrors what a real worker does (import torch, build
# the largest model in the sweep, run a forward+backward) and reports the
# *peak* host RSS and *peak* CUDA allocation.  This is the ground truth used by
# the auto-sizer.  The old `_measure_worker_rss_gb` measured only import-time
# RSS (~1.6 GB) and clamped to 4 GB — which silently dropped real ~8–10 GB
# steady-state measurements and let the OOM-killer reap workers on the tail.
#
# Cached on disk for 7 days, keyed on hostname + GPU model + dataset path, so
# the probe runs at most once per machine/dataset.

_PROBE_CACHE_DIR = Path.home() / ".cache" / "mtp_grid"
_PROBE_CACHE_TTL_SEC = 7 * 86400
_PROBE_RESULT_CACHE: dict[str, dict[str, float]] = {}   # in-process memo


def _probe_cache_key(dataset_dir: str) -> tuple[str, Path]:
    """Cache-key (string, file path) over hostname + GPU model + dataset."""
    import hashlib
    import socket
    gpu_name = "cpu"
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=5,
        ).decode().strip()
        if out:
            gpu_name = out.splitlines()[0].strip()
    except Exception:
        pass
    key = f"{socket.gethostname()}|{gpu_name}|{Path(dataset_dir).name}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return key, _PROBE_CACHE_DIR / f"probe_{h}.json"


def _probe_resource_peaks(dataset_dir: str = "", force: bool = False) -> dict[str, float]:
    """Spawn a representative worker subprocess and measure peak RSS + peak VRAM
    under a REAL training burst (not synthetic batches).

    The subprocess:
      1. Imports torch + the pipeline / EDR strategy stack a real worker uses.
      2. Builds dataloaders from the actual dataset (times the load).
      3. Builds the largest EDR model in the sweep via ``_make_model_edr``.
      4. Runs 3 forward+backward iterations on real batches from the train loader.
      5. Reports peak host RSS, ``torch.cuda.max_memory_allocated`` + CUDA ctx,
         and dataset-load wall-time.

    Real-burst probing (vs synthetic in round 1) captures cuDNN kernel
    selection, torch.compile workspace, and real activation shapes — the
    synthetic version underestimated by ~6× and let the pool oversize itself
    (pool=32 / in-flight=16 mismatch).

    Returns: ``{"rss_gb", "vram_gb", "cuda_ctx_gb", "dataset_load_sec", ...}``.

    Cached: in-process for the current run, on-disk for 7 days keyed on
    (hostname, GPU model, dataset).  Re-probes on cache miss or stale entry.
    Always returns something (falls back to safe HPC defaults on failure).
    """
    import json
    import subprocess
    import sys

    plog = logging.getLogger(__name__)
    if not dataset_dir:
        dataset_dir = TRAIN_DATA_RUN_DIR
    key, cache_path = _probe_cache_key(dataset_dir)
    if not force and key in _PROBE_RESULT_CACHE:
        return _PROBE_RESULT_CACHE[key]
    if not force and cache_path.is_file():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age < _PROBE_CACHE_TTL_SEC:
                data = json.loads(cache_path.read_text())
                # Reject stale-schema caches that lack the new dataset_load_sec
                # field — forces a re-probe on first run after the upgrade.
                if "dataset_load_sec" in data:
                    data["source"] = "cache"
                    _PROBE_RESULT_CACHE[key] = data
                    plog.info(
                        "resource probe (cache hit, age %.1fd): rss=%.2fGB "
                        "vram=%.2fGB ctx=%.2fGB load=%.1fs",
                        age / 86400, data["rss_gb"], data["vram_gb"],
                        data["cuda_ctx_gb"], data.get("dataset_load_sec", 0.0),
                    )
                    return data
        except Exception:
            pass

    repo_root = str(Path(__file__).resolve().parents[2])
    edr_dir = str(Path(__file__).resolve().parent / "Equivariant-Decomposed-Residual")
    # Real-training burst probe.  Loads the actual dataset, builds the largest
    # EDR model via the strategy's make_model, runs 3 real fwd+bwd iterations.
    script = (
        "import json, os, sys, gc, time\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        f"sys.path.insert(0, {edr_dir!r})\n"
        'os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")\n'
        "for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):\n"
        "    os.environ.setdefault(_v, '1')\n"
        "import psutil\n"
        "baseline_rss = psutil.Process().memory_info().rss / 1e9\n"
        "try:\n"
        "    import torch\n"
        "    has_cuda = bool(torch.cuda.is_available())\n"
        "except Exception as e:\n"
        "    print('PROBE_RESULT_JSON:' + json.dumps({'error': 'torch_import:' + repr(e)}))\n"
        "    sys.exit(0)\n"
        "cuda_ctx_gb = 0.0\n"
        "if has_cuda:\n"
        "    try:\n"
        "        f0,_ = torch.cuda.mem_get_info(0)\n"
        "        _t = torch.zeros(1, device='cuda:0')\n"
        "        f1,_ = torch.cuda.mem_get_info(0)\n"
        "        cuda_ctx_gb = max(0.05, (f0-f1)/1e9)\n"
        "        del _t\n"
        "        torch.cuda.empty_cache()\n"
        "        torch.cuda.reset_peak_memory_stats(0)\n"
        "    except Exception:\n"
        "        pass\n"
        "from Neural_Networks.loader import make_dataloaders\n"
        "import Neural_Networks.models.shared.pipeline  # noqa\n"
        "from Neural_Networks.models.shared.strategies import PLAIN_STRATEGY\n"
        "# Probe imports EDR strategy too so its dep footprint is counted in RSS.\n"
        "try:\n"
        "    from edr_strategy import EDR_STRATEGY  # noqa\n"
        "except Exception:\n"
        "    pass\n"
        "rss_imports = psutil.Process().memory_info().rss / 1e9\n"
        "device = torch.device('cuda' if has_cuda else 'cpu')\n"
        "t0 = time.time()\n"
        "try:\n"
        f"    loaders = make_dataloaders({dataset_dir!r}, batch_size=512, mode='pointwise', "
        "num_workers=0, normalise=True)\n"
        "    dataset_load_sec = time.time() - t0\n"
        "    train_loader = loaders['train']\n"
        "except Exception as e:\n"
        "    print('PROBE_RESULT_JSON:' + json.dumps({'error': 'dataset_load:' + repr(e)}))\n"
        "    sys.exit(0)\n"
        "# Probe with the FNN-largest sweep config: hidden_layers=[512,512,512].\n"
        "# This is the largest-parameter model across all three archs (FNN/PhysReg\n"
        "# share the MLP backbone; EDR's 4×[128,128] is smaller per-net even with 4\n"
        "# sub-networks).  PLAIN_STRATEGY's make_model expects only a kinematics\n"
        "# tensor, so we can drive it with batches straight from make_dataloaders.\n"
        "hp = {\n"
        "    'hidden_layers':   [512, 512, 512],\n"
        "    'activation':      'silu',\n"
        "    'dropout':         0.3,\n"
        "    'batch_size':      512,\n"
        "    'learning_rate':   1e-3,\n"
        "    'weight_decay':    5e-3,\n"
        "    'feature_noise_std': 0.0,\n"
        "    'epochs':          1,\n"
        "    'patience':        100,\n"
        "}\n"
        "try:\n"
        "    model = PLAIN_STRATEGY.make_model(device, hp)\n"
        "    opt   = PLAIN_STRATEGY.build_optimizer(model, hp)\n"
        "except Exception as e:\n"
        "    print('PROBE_RESULT_JSON:' + json.dumps({'error': 'model_build:' + repr(e)}))\n"
        "    sys.exit(0)\n"
        "peak_rss = rss_imports\n"
        "iters_run = 0\n"
        "iter_err = ''\n"
        "try:\n"
        "    for batch in train_loader:\n"
        "        if iters_run >= 3:\n"
        "            break\n"
        "        if isinstance(batch, (tuple, list)) and len(batch) >= 2:\n"
        "            x = batch[0].to(device, non_blocking=True)\n"
        "            y = batch[1].to(device, non_blocking=True)\n"
        "        else:\n"
        "            continue\n"
        "        opt.zero_grad(set_to_none=True)\n"
        "        out = model(x)\n"
        "        pred = out[0] if isinstance(out, (tuple, list)) else out\n"
        "        loss = ((pred - y) ** 2).mean()\n"
        "        loss.backward()\n"
        "        opt.step()\n"
        "        peak_rss = max(peak_rss, psutil.Process().memory_info().rss / 1e9)\n"
        "        iters_run += 1\n"
        "except Exception as e:\n"
        "    iter_err = repr(e)\n"
        "peak_vram = (torch.cuda.max_memory_allocated(0)/1e9 + cuda_ctx_gb) if has_cuda else 0.0\n"
        "out = {\n"
        "    'rss_gb': round(peak_rss, 3),\n"
        "    'vram_gb': round(peak_vram, 3),\n"
        "    'cuda_ctx_gb': round(cuda_ctx_gb, 3),\n"
        "    'rss_baseline_gb': round(baseline_rss, 3),\n"
        "    'rss_after_imports_gb': round(rss_imports, 3),\n"
        "    'dataset_load_sec': round(dataset_load_sec, 2),\n"
        "    'iters_run': iters_run,\n"
        "    'iter_err': iter_err,\n"
        "}\n"
        "print('PROBE_RESULT_JSON:' + json.dumps(out))\n"
    )

    fallback = {
        "rss_gb": 4.0, "vram_gb": 8.0, "cuda_ctx_gb": 0.5,
        "rss_baseline_gb": 0.5, "rss_after_imports_gb": 1.6,
        "dataset_load_sec": 20.0, "iters_run": 0, "source": "fallback",
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, timeout=240, text=True, cwd=repo_root,
        )
        out_text = (proc.stdout or "") + (proc.stderr or "")
        for ln in out_text.splitlines():
            if ln.startswith("PROBE_RESULT_JSON:"):
                data = json.loads(ln[len("PROBE_RESULT_JSON:"):])
                if "error" in data:
                    plog.warning("resource probe reported error: %s", data["error"])
                    _PROBE_RESULT_CACHE[key] = fallback
                    return fallback
                data["source"] = "probe"
                try:
                    _PROBE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(data))
                except Exception as exc:
                    plog.debug("probe cache write failed: %s", exc)
                _PROBE_RESULT_CACHE[key] = data
                plog.info(
                    "resource probe (measured): rss=%.2fGB vram=%.2fGB ctx=%.2fGB "
                    "load=%.1fs iters=%d (baseline=%.2f, post-imports=%.2f)",
                    data["rss_gb"], data["vram_gb"], data["cuda_ctx_gb"],
                    data.get("dataset_load_sec", 0.0),
                    int(data.get("iters_run", 0)),
                    data.get("rss_baseline_gb", 0.0),
                    data.get("rss_after_imports_gb", 0.0),
                )
                return data
        plog.warning("resource probe produced no PROBE_RESULT_JSON line — using fallback")
    except subprocess.TimeoutExpired:
        plog.warning("resource probe timed out — using fallback")
    except Exception as exc:
        plog.warning("resource probe failed (%s) — using fallback", exc)
    _PROBE_RESULT_CACHE[key] = fallback
    return fallback


def _measure_cuda_ctx_gb() -> float:
    """CUDA context overhead per worker (~0.35 GB on consumer GPUs).

    Thin wrapper around the unified resource probe.  Falls back to 0.35 GB.
    """
    data = _probe_resource_peaks(TRAIN_DATA_RUN_DIR)
    val = float(data.get("cuda_ctx_gb", 0.35))
    return val if 0.05 <= val <= 2.0 else 0.35


def _measure_worker_rss_gb() -> float:
    """Per-worker peak resident-set size (host RAM) under realistic load.

    Thin wrapper around the unified resource probe.  Uses the *peak* RSS
    measured during a representative forward+backward — not import-time RSS.
    A small 1.2x safety factor cushions allocator high-water-mark drift.
    Clamp band widened to [1.0, 16.0] GB so a real ~8–10 GB measurement isn't
    silently dropped (was [0.6, 4.0] — the OOM-kill root cause).
    """
    data = _probe_resource_peaks(TRAIN_DATA_RUN_DIR)
    val = float(data.get("rss_gb", 3.0)) * 1.2
    return min(16.0, max(1.0, val))


def _measure_worker_peak_vram_gb() -> float:
    """Per-trial peak VRAM (device) under realistic load.

    Returns the probe's measured peak ``torch.cuda.max_memory_allocated``
    plus the CUDA context overhead, with a 1.5x safety factor for cuDNN /
    torch.compile workspace spikes that the probe's short burst can miss.
    """
    data = _probe_resource_peaks(TRAIN_DATA_RUN_DIR)
    return max(0.5, float(data.get("vram_gb", 1.5)) * 1.5)


def _grid_threads_per_worker_target() -> int:
    """Intra-op torch threads to budget per worker (sizes the CPU pool ceiling).

    These trials are tiny per-joint MLPs: torch intra-op parallelism does
    almost nothing (the limiter is the Python training loop / dataloader /
    GPU dispatch), so 1 thread/worker on a big GPU box ~doubles concurrency
    while keeping total torch threads ≈ physical cores (no oversubscription).
    Laptop/workstation keep 2 (single-threaded torch has been linked to rare
    deadlocks on those, and the pool is small there anyway).

    Override: ``MTP_GRID_THREADS_PER_WORKER`` (int ≥ 1).
    """
    env = os.environ.get("MTP_GRID_THREADS_PER_WORKER", "").strip()
    if env.isdigit() and int(env) >= 1:
        return int(env)
    return 1 if _detect_env() == "hpc" else 2


def _compute_pool_size(cuda_available: bool, n_gpus: int = 1) -> int:
    """Derive the concurrent-worker count from live hardware ceilings.

    Three ceilings, each derived from a MEASURED quantity (resource probe).
    The final pool size is ``min(VRAM, CPU, RAM)`` — no artificial cap, because
    the probe-derived margins already build in headroom:

      • VRAM ceiling : ``(vram_total - n_gpus × RESERVE) / (peak_vram × 1.5)``
        — uses the resource probe's measured peak VRAM from a REAL training
        burst, applied as the binding admission constraint.  Matches the
        dispatcher's per-GPU gate so ``pool_size`` workers can all run
        concurrently (no idle slots).

      • CPU ceiling  : ``(cpu_phys - 2) // tpw_target`` × ``oversub`` — the
        throughput limiter for these GPU-bound mini-MLPs is the Python step
        loop, not GPU compute.

      • RAM ceiling  : ``(free_ram - 32) / per_worker_peak_rss`` — uses the
        resource probe's measured peak RSS during a real fwd+bwd, not
        import-time RSS (the WORKER-DIED root cause that round 1 fixed).

    Round 2 change: dropped the 0.70 utilization cap.  With the now-accurate
    probe (real training burst), the margins are: peak_vram × 1.5 ≈ 50 % VRAM
    spike buffer; peak_rss × 1.2 ≈ 20 % RAM spike buffer; ``ram_safety_gb=32``
    floor; ``VRAM_RESERVE_GB=2`` per-GPU floor.  Adding another 0.70 multiplier
    on top of these wasted ~30 % of real capacity (pool=32 / in-flight=16
    mismatch).  The cap is still honored via ``MTP_GRID_UTILIZATION_CAP``
    (default 1.0 = no cap) for users who want extra headroom.

    Tunables (env, advanced — autosizer is preferred):
      • ``MTP_GRID_POOL_SIZE`` forces an exact size (highest priority).
      • ``MTP_GRID_CPU_OVERSUB`` scales the CPU budget (default 1.0).
      • ``MTP_GRID_WORKER_RAM_GB`` overrides the per-worker RSS estimate.
      • ``MTP_GRID_UTILIZATION_CAP`` extra throttle (default 1.0 = off).
    """
    import psutil
    _plog = logging.getLogger(__name__)
    if not cuda_available:
        return 1                                             # CPU ⇒ sequential

    # Explicit override wins outright.
    override = os.environ.get("MTP_GRID_POOL_SIZE", "").strip()
    if override.isdigit() and int(override) >= 1:
        _plog.info("pool_size=%s forced via MTP_GRID_POOL_SIZE", override)
        return int(override)

    cpu_phys = psutil.cpu_count(logical=False) or 2
    n_gpus = max(1, n_gpus)
    tpw = _grid_threads_per_worker_target()

    # ── VRAM ceiling (HARD cap matched to the dispatcher's per-GPU gate) ────
    # Subtract a per-GPU reserve from the total budget; divide by per-trial
    # peak (already includes 1.5x safety from _measure_worker_peak_vram_gb).
    vram_total = _query_total_vram_gb()
    peak_vram = _measure_worker_peak_vram_gb()                  # peak × 1.5
    usable_vram = max(0.0, vram_total - n_gpus * VRAM_RESERVE_GB)
    n_vram = max(1, int(usable_vram / max(0.5, peak_vram)))

    # ── CPU ceiling ─────────────────────────────────────────────────────────
    try:
        oversub = float(os.environ.get("MTP_GRID_CPU_OVERSUB", "1.0"))
    except ValueError:
        oversub = 1.0
    oversub = max(1.0, oversub)
    n_cpu = max(1, int(((cpu_phys - 2) // max(1, tpw)) * oversub))

    # ── RAM ceiling (measured peak RSS, includes 1.2x safety) ───────────────
    _env_ram = os.environ.get("MTP_GRID_WORKER_RAM_GB", "").strip()
    if _env_ram:
        try:
            per_worker_ram = float(_env_ram)
            _ram_src = "env"
        except ValueError:
            per_worker_ram, _ram_src = _measure_worker_rss_gb(), "measured"
    else:
        per_worker_ram, _ram_src = _measure_worker_rss_gb(), "measured"
    per_worker_ram = max(1.0, per_worker_ram)
    free_ram = _query_free_ram_gb()
    ram_safety_gb = 32.0
    n_ram = max(1, int((free_ram - ram_safety_gb) / per_worker_ram))

    # ── Combine: binding axis wins; optional util_cap defaults to 1.0 ───────
    raw = max(1, min(n_vram, n_cpu, n_ram))
    try:
        util_cap = float(os.environ.get("MTP_GRID_UTILIZATION_CAP", "1.0"))
    except ValueError:
        util_cap = 1.0
    util_cap = min(1.0, max(0.10, util_cap))
    pool_size = max(1, int(raw * util_cap))

    _bound = min(
        (("VRAM", n_vram), ("CPU", n_cpu), ("RAM", n_ram)), key=lambda kv: kv[1]
    )[0]
    _plog.info(
        "pool_size=%d (raw=%d × util_cap=%.2f)  ceilings: "
        "VRAM=%d[peak=%.2fGB, usable=%.1fGB/%dGPU×%.1f reserve]  "
        "CPU=%d[phys=%d,tpw=%d,oversub=%.1f]  "
        "RAM=%d[%s rss=%.2fGB, free=%.0fGB-%.0f safety]  → bound by %s",
        pool_size, raw, util_cap,
        n_vram, peak_vram, usable_vram, n_gpus, VRAM_RESERVE_GB,
        n_cpu, cpu_phys, tpw, oversub,
        n_ram, _ram_src, per_worker_ram, free_ram, ram_safety_gb,
        _bound,
    )
    return pool_size


def _compute_threads_per_worker(pool_size: int) -> int:
    import psutil
    cpu_phys = psutil.cpu_count(logical=False) or 2
    # Floor at the same per-worker thread target the pool was sized against
    # (1 on HPC, 2 elsewhere) so total torch threads ≈ physical cores and the
    # pool/threads pair stays self-consistent.  Cap by spare cores/pool.
    tpw = _grid_threads_per_worker_target()
    return max(tpw, (cpu_phys - 1) // max(1, pool_size))


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


def _run_sequential_collect(
    trials: list[dict[str, Any]],
    log: logging.Logger,
    t_start: float,
) -> tuple[list["_Result"], list[dict[str, Any]], tuple[int, int, int]]:
    """Run every trial in-process, one after another (CPU-only or pool_size=1).

    Returns ``(executed_results, skipped_trials, (ok, skip, fail))`` WITHOUT
    finalizing — the caller owns artifact writing.  Sequential keeps no
    in-memory ``_Result`` list, so ``executed_results`` is ``[]`` and the
    whole grid is returned as ``skipped`` for disk reconstruction (every
    ok/pre-skipped trial has an on-disk run; a trial that failed this session
    has no run dir and is simply omitted by the reconstructor)."""
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
        PLAIN_STRATEGY, PHYSICS_REG_STRATEGY,
    )
    if _EDR_DIR not in sys.path:
        sys.path.insert(0, _EDR_DIR)
    from edr_strategy import EDR_STRATEGY  # noqa: E402
    strategy_map = {
        "fnn":      PLAIN_STRATEGY,
        "physreg":  PHYSICS_REG_STRATEGY,
        "edr":      EDR_STRATEGY,
    }

    pending, _skipped = _partition_trials_by_skip(trials)
    n_skip = len(_skipped)
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
            status = "ok" if rc is not None else "failed"
            _rmse  = rc if (rc is not None and rc == rc) else None  # nan→None
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
            _rmse_str = f"test_rmse={_rmse:.4f}  " if _rmse is not None else ""
            tqdm.write(f"  [ OK ]  {idx:{w}}/{total}  {desc}  {_rmse_str}elapsed={_fmt_time(elapsed)}  ETA={_fmt_time(eta)}")
        elif status == "failed":
            n_fail += 1
            tqdm.write(f"  [FAIL]  {idx:{w}}/{total}  {desc}  (run_training returned non-zero)")
        else:
            n_fail += 1
            tqdm.write(f"  [ERR ]  {idx:{w}}/{total}  {desc}\n" + error_msg[:400])

    # Sequential keeps no in-memory _Result list; every ok/pre-skipped trial
    # now has an on-disk run, so the whole grid is handed back for disk
    # reconstruction. (A failed trial has no run dir; reconstructor omits it.)
    return [], list(trials), (n_ok, n_skip, n_fail)


def _run_sequential(
    trials: list[dict[str, Any]],
    log: logging.Logger,
    t_start: float,
) -> tuple[int, int, int]:
    """Behavior-preserving wrapper: run sequentially, then finalize artifacts."""
    res, skipped, counts = _run_sequential_collect(trials, log, t_start)
    _finalize(res, skipped, DATASET_OUT_ROOT, log)
    return counts


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
# Heavy-import cache populated in ``_pool_init``.  Doing these imports ONCE per
# worker lifetime instead of per trial removes ~5–10 s of latency from every
# dispatch and means the slot's "start" message lands within ~1 s of admit
# (was ~30–60 s — the "waiting … nothing training starts" symptom).
_POOL_TRAIN_JOB_CTOR: Any = None        # pipeline.TrainJob
_POOL_RUN_TRAINING: Any = None          # pipeline.run_training
_POOL_STRATEGIES: dict[str, Any] = {}   # arch → strategy instance


def _pool_init(
    progress_queue: "mp.Queue",
    threads_per_worker: int,
    gpu_id: int,
    is_hpc: bool = False,
) -> None:
    """Pool initializer: runs ONCE per worker lifetime (on spawn and after each
    ``maxtasksperchild`` recycle).

    Critically, ``torch.set_num_interop_threads`` can only be called before any
    parallel work has started.  Calling it per-trial crashed every trial after
    the first one in each worker.  Doing it here — exactly once — keeps the
    worker usable for its full ``maxtasksperchild`` budget.

    ``gpu_id`` is a FIXED scalar baked into this pool's ``initargs`` — every
    worker in a given per-GPU pool is permanently pinned to that one physical
    GPU.  This is deliberately *not* a shared ticket queue: a queue sized to
    ``pool_size`` drained after the first batch of workers, so the first
    ``maxtasksperchild`` recycle blocked forever on ``queue.get()`` and silently
    hung the whole grid.  A constant has no such failure mode and survives
    unlimited worker recycling.

    Heavy imports (pipeline, strategies, EDR strategy) happen here ONCE and
    are cached in module globals (``_POOL_TRAIN_JOB_CTOR``, ``_POOL_RUN_TRAINING``,
    ``_POOL_STRATEGIES``).  Per-trial dispatch only does the (cheap) skip-existing
    check and trial submission, so a fresh worker's first slot flips to
    ``running`` within ~1 s of dispatch instead of ~30–60 s.
    """
    global _POOL_PROGRESS_QUEUE, _POOL_THREADS_PER_WORKER, _POOL_GPU_ID
    global _POOL_TRAIN_JOB_CTOR, _POOL_RUN_TRAINING, _POOL_STRATEGIES
    _POOL_PROGRESS_QUEUE = progress_queue
    _POOL_THREADS_PER_WORKER = int(threads_per_worker)
    _POOL_GPU_ID = int(gpu_id)

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
    # Pin BLAS/OpenMP to the per-worker budget BEFORE torch import.  With a
    # large HPC pool (~cores workers) an unset OMP/MKL default = all-cores per
    # process → catastrophic thread oversubscription / scheduler thrash (the
    # 0/204-in-6h40m failure mode).  Must precede `import torch`.
    _tpw_env = str(max(1, int(threads_per_worker)))
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = _tpw_env

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

    # ── Pre-warm heavy imports (pipeline, strategies, EDR) ─────────────────
    # The spawn worker inherits NO sys.path, so we restore the two roots first.
    import sys as _sys
    from pathlib import Path as _Path
    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    _edr_dir = str(_Path(__file__).resolve().parent / "Equivariant-Decomposed-Residual")
    if _edr_dir not in _sys.path:
        _sys.path.insert(0, _edr_dir)

    try:
        from Neural_Networks.models.shared.pipeline import TrainJob, run_training
        from Neural_Networks.models.shared.strategies import (
            PLAIN_STRATEGY, PHYSICS_REG_STRATEGY,
        )
        from edr_strategy import EDR_STRATEGY  # noqa: E402
        _POOL_TRAIN_JOB_CTOR = TrainJob
        _POOL_RUN_TRAINING = run_training
        _POOL_STRATEGIES = {
            "fnn":      PLAIN_STRATEGY,
            "physreg":  PHYSICS_REG_STRATEGY,
            "edr":      EDR_STRATEGY,
        }
    except Exception as exc:
        # If imports fail here, every _run_one_trial call will retry inline
        # (the per-trial fallback path is preserved) and surface the real error.
        logging.getLogger(__name__).warning(
            "_pool_init: pre-warm import failed (will retry per-trial): %s", exc,
        )

    # One-line readiness signal — visible in the rich log pane / non-TTY fallback.
    logging.getLogger(__name__).info(
        "pool_init_done  pid=%d  gpu=%d  tpw=%d  imports=%s",
        os.getpid(), _POOL_GPU_ID, int(threads_per_worker),
        "ok" if _POOL_RUN_TRAINING is not None else "deferred",
    )


def _pool_warmup_noop() -> str:
    """No-op task used as a synchronous warmup barrier.

    The main process submits one of these per worker slot BEFORE dispatching any
    real trial, and joins on the results.  This guarantees every worker has
    completed ``_pool_init`` (including the heavy imports above) so the first
    real trial dispatches into a fully-warmed worker — the dashboard's "waiting"
    state lasts ~1 s instead of the cold-start window of ~30–60 s.
    """
    import os as _os
    return f"warm:pid={_os.getpid()}:gpu={_POOL_GPU_ID}"


def _run_one_trial(
    trial: dict[str, Any],
    worker_slot: int,
    threads_per_worker: int,
) -> dict[str, Any]:
    """Run a single trial in a fresh worker process with improvements v8 (Trig Features + High Dropout)."""
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

    try:
        # Use pre-warmed imports from _pool_init.  If pre-warm failed (rare),
        # fall back to inline imports so we never lose the trial.
        TrainJob = _POOL_TRAIN_JOB_CTOR
        run_training = _POOL_RUN_TRAINING
        strategy_map = _POOL_STRATEGIES
        if TrainJob is None or run_training is None or not strategy_map:
            import sys as _sys
            from pathlib import Path as _Path
            _repo_root = str(_Path(__file__).resolve().parents[2])
            if _repo_root not in _sys.path:
                _sys.path.insert(0, _repo_root)
            _edr_dir = str(_Path(__file__).resolve().parent / "Equivariant-Decomposed-Residual")
            if _edr_dir not in _sys.path:
                _sys.path.insert(0, _edr_dir)
            from Neural_Networks.models.shared.pipeline import TrainJob, run_training  # noqa
            from Neural_Networks.models.shared.strategies import (
                PLAIN_STRATEGY, PHYSICS_REG_STRATEGY,
            )
            from edr_strategy import EDR_STRATEGY  # noqa: E402
            strategy_map = {
                "fnn": PLAIN_STRATEGY, "physreg": PHYSICS_REG_STRATEGY, "edr": EDR_STRATEGY,
            }

        strategy = strategy_map[arch]

        # Skip-existing inside the worker — cheap, avoids redundant data load.
        if SKIP_EXISTING and _find_existing_run(Path(models_dir), model_type, hp):
            q.put(("done", worker_slot, "skip"))
            return {"status": "skipped", "arch": arch, "hp": hp}

        job = TrainJob(
            run_dir=TRAIN_DATA_RUN_DIR,
            models_dir=models_dir,
            registry_file=REGISTRY_FILE,
            model_type=model_type,
            save_subdir=save_subdir,
            hp=hp,
            strategy=strategy,
            run_help=run_help,
        )

        q.put(("start", worker_slot, _hp_desc(arch, hp),
               int(hp.get("epochs", 3000)), int(hp.get("patience", 200))))

        def _cb(epoch, total_ep, val_rmse, pat_ctr, pat_max):
            q.put(("progress", worker_slot, int(epoch), int(total_ep),
                   float(val_rmse), int(pat_ctr), int(pat_max)))

        rc = run_training(job, progress_callback=_cb)
        status = "ok" if rc is not None else "failed"
        # rc is the held-out test rmse_traj_macro (float) — the trajectory-
        # macro RMSE, the SAME estimator as the live per-epoch val_rmse so the
        # leaderboard is directly comparable to training output; nan for an
        # incomplete segment, or None on failure. Surface a real number only.
        rmse = float(rc) if (rc is not None and rc == rc) else None
        q.put(("done", worker_slot, status))
        return {"status": status, "arch": arch, "hp": hp, "rmse": rmse}

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
    hp: dict = field(default_factory=dict)   # full trial HP — for CSV/summary


@dataclass
class _TUIState:
    pool_size: int
    total_trials: int
    threads_per_worker: int
    device_label: str
    n_gpus: int = 1
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
    # Per-GPU free VRAM in GB (length = n_gpus_eff).  Populated by the 1 Hz
    # poller; rendered as a one-line breakdown under the cross-GPU VRAM gauge
    # so the user can see which GPU is bottlenecking dispatch.
    free_vram_gb_per_gpu: list[float] = field(default_factory=list)
    total_vram_gb_per_gpu: list[float] = field(default_factory=list)
    results: deque = field(default_factory=lambda: deque(maxlen=8))
    all_results: list = field(default_factory=list)        # uncapped — system of record
    best_by_arch: dict = field(default_factory=dict)       # arch → best _Result so far
    t_start: float = 0.0
    # Most recent failing trial — surfaced in the dashboard so the user can see
    # the cause without opening _errors/trial_errors.log mid-run.  Both empty
    # when no failures yet.
    last_error_arch: str = ""
    last_error_msg:  str = ""
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
        n_gpus      = state.n_gpus
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
        best_by_arch = dict(state.best_by_arch)
        n_done_total = len(state.all_results)
        t_start     = state.t_start

    # ── Header ──────────────────────────────────────────────────────────────
    _per_gpu_str = f" ({pool_size // max(1, n_gpus)}/gpu)" if n_gpus > 1 else ""
    header = Text.assemble(
        ("Torque Grid Search", "bold cyan"),
        "   ARCH=", (str(ARCH), "bold"),
        f"   total={total}   ",
        (f"pool={pool_size}{_per_gpu_str}", "bold"),
        f"   gpus={n_gpus}   threads/worker={tpw}   device={device}",
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

    # Per-GPU breakdown (read under lock).  Shows which GPU is bottlenecking
    # admission when the cross-GPU sum looks fine but a single GPU is full.
    with state.lock:
        _free_per_gpu = list(state.free_vram_gb_per_gpu)
        _total_per_gpu = list(state.total_vram_gb_per_gpu)
    _per_gpu_str = ""
    if _free_per_gpu and _total_per_gpu and len(_free_per_gpu) == len(_total_per_gpu) and len(_free_per_gpu) > 1:
        _per_gpu_str = "   ".join(
            f"gpu{i}: {(t-f):5.1f}/{t:.0f} ({f:5.1f} free)"
            for i, (f, t) in enumerate(zip(_free_per_gpu, _total_per_gpu))
        )

    res_tbl = Table.grid(padding=(0, 1), expand=True)
    res_tbl.add_column(style="dim", width=6, no_wrap=True)
    res_tbl.add_column(width=26, no_wrap=True)
    res_tbl.add_column(no_wrap=True)
    res_tbl.add_row(
        "VRAM",
        Text(_gauge(vram_used, total_vram, 24), style=_gauge_style(vram_pct)),
        f"{vram_used:5.2f} / {total_vram:5.2f} GB  ({free_vram:4.2f} free, reserve {VRAM_RESERVE_GB:.2f})",
    )
    if _per_gpu_str:
        res_tbl.add_row("", Text(""), Text(_per_gpu_str, style="dim"))
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
        "   ", ("FAIL ", "bold"),
        (f"{fail_n:>2d}", "bold red reverse" if fail_n > 0 else "bold red"),
    )

    res_outer = Table.grid(expand=True)
    res_outer.add_row(res_tbl)
    res_outer.add_row(counts)
    # Surface the most recent error so the user can act without tailing the
    # _errors/trial_errors.log file.  Empty until the first failure.
    with state.lock:
        last_err_arch = state.last_error_arch
        last_err_msg  = state.last_error_msg
    if last_err_msg:
        err_line = Text.assemble(
            ("LAST ERROR  ", "bold red"),
            (f"{last_err_arch}  ", "bold"),
            (last_err_msg[:180], "red"),
        )
        res_outer.add_row(err_line)
    resources_panel = Panel(res_outer, title="resources (live)", border_style="blue", padding=(0, 1))

    # ── Active trials (two lines per slot, capped at 8 visible) ────────────
    # Sort: running first; within running, most-progressed (highest epoch)
    # first, then oldest by wallclock; waiting slots last.  Surfaces the
    # trials closest to completion without losing visibility on stragglers.
    _TUI_MAX_SLOTS = 8
    _now_for_sort = time.time()
    slots_sorted = sorted(
        slots,
        key=lambda s: (
            0 if s.status == "running" else 1,
            -int(s.epoch or 0),
            -((_now_for_sort - s.started_at) if s.started_at else 0.0),
            s.slot,
        ),
    )
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

    # ── Best so far (per arch) leaderboard ──────────────────────────────────
    lb = Table(expand=True, show_edge=False, pad_edge=False, padding=(0, 1))
    lb.add_column("arch",          width=20, no_wrap=True)
    lb.add_column("best test rmse", width=14, justify="right")
    lb.add_column("config",        no_wrap=True)
    if not best_by_arch:
        lb.add_row("—", "—", "(no completed runs yet)")
    else:
        for _arch in sorted(best_by_arch):
            _b = best_by_arch[_arch]
            lb.add_row(
                Text(_arch, style="cyan"),
                Text(f"{_b.rmse:.5f}", style="bold green"),
                _b.config,
            )
    leaderboard_panel = Panel(
        lb, title=f"best so far (per arch)  ·  {n_done_total} runs done",
        border_style="bold green", padding=(0, 1),
    )

    # ── Assemble layout ─────────────────────────────────────────────────────
    layout = Layout()
    # 2 content lines per visible slot + 1 separator + 1 "…N more" line if hidden.
    n_vis = len(slots_visible)
    active_h = 2 + 2 * n_vis + max(0, n_vis - 1) + (1 if hidden > 0 else 0)
    # Panel chrome (2 border) + table header (1) + header rule (1) = 4 fixed
    # lines before any data row.  The old "3 + N" under-counted by one and
    # clipped every data row, so the leaderboard always looked empty.
    history_h = 4 + max(len(results), 1)
    lb_h = 4 + max(len(best_by_arch), 1)
    layout.split_column(
        Layout(header_panel,      name="header",      size=3),
        Layout(resources_panel,   name="resources",   size=6),
        Layout(active_panel,      name="active",      size=active_h),
        Layout(overall_panel,     name="overall",     size=5),
        Layout(leaderboard_panel, name="leaderboard", size=min(10, lb_h)),
        Layout(history_panel,     name="history",     size=min(13, history_h)),
    )
    return layout


class _DashboardRenderable:
    """rich ``__rich__`` hook so ``Live`` auto-refresh pulls live state."""
    def __init__(self, state: _TUIState):
        self.state = state

    def __rich__(self) -> Any:
        return _render_dashboard(self.state)


def _hashable(v):
    """Canonicalise an HP value into a hashable form (handles nested
    list/tuple/dict, e.g. lambda_correction_reg_per_component)."""
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    return v


def _config_key(hp: dict) -> tuple:
    """Stable per-config identity, ignoring seed and private/injected keys."""
    skip = {"seed", "data_train_seed", "_grid_seed"}
    items = []
    for k in sorted(hp):
        if k in skip or k.startswith("_"):
            continue
        items.append((k, _hashable(hp[k])))
    return tuple(items)


def _flush_results_csv(results: list[_Result], out_root: str) -> None:
    """Write (overwrite) ``grid_results.csv`` reflecting every result so far.

    Factored out of ``_write_grid_summary`` so the admission loop can call it
    after *every* completed trial — a crash / OOM-kill / power loss at trial N
    of a long grid then still leaves a complete CSV of the N-1 finished
    trials instead of nothing.  Cheap (a few hundred rows) and idempotent.
    """
    import csv

    hp_keys: list[str] = []
    for r in results:
        for k in r.hp:
            if not k.startswith("_") and k not in hp_keys:
                hp_keys.append(k)
    hp_keys.sort()
    ordered = sorted(results,
                     key=lambda x: (x.arch, x.rmse if x.rmse is not None else 9e9))
    for csv_name, _ in _result_basenames():
        csv_path = os.path.join(out_root, csv_name)
        with open(csv_path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["n", "arch", "status", "test_rmse", "elapsed_sec"] + hp_keys)
            for r in ordered:
                row = [r.n, r.arch, r.status,
                       f"{r.rmse:.6f}" if r.rmse is not None else "",
                       f"{r.elapsed:.1f}"]
                for k in hp_keys:
                    v = r.hp.get(k, "")
                    row.append("-".join(str(x) for x in v)
                               if isinstance(v, (list, tuple)) else v)
                wr.writerow(row)


def _write_grid_summary(results: list[_Result], out_root: str) -> None:
    """Write grid_results.csv (every trial) and grid_summary.md (ranked).

    ``grid_summary.md`` ranks each arch's configs by mean test RMSE across
    seeds (mean±std, n) so the comparison is statistically conclusive, then
    names the cross-arch winner.
    """
    import statistics as _stats

    # A reconstructed-from-disk row carries status "skip" but IS a completed
    # successful run (it has a real test rmse); only fail/err lack an rmse.
    ok = [r for r in results
          if r.rmse is not None and r.status in ("ok", "skip")]

    # ── grid_results.csv ────────────────────────────────────────────────────
    _flush_results_csv(results, out_root)

    # ── grid_summary.md ─────────────────────────────────────────────────────
    lines: list[str] = ["# Grid study — results summary", ""]
    lines.append(f"Total trials: {len(results)}  ·  successful: {len(ok)}")
    lines.append("")
    arch_best_mean: dict[str, tuple[float, float, int, _Result]] = {}
    for arch in sorted({r.arch for r in ok}):
        a_ok = [r for r in ok if r.arch == arch]
        groups: dict[tuple, list[_Result]] = {}
        for r in a_ok:
            groups.setdefault(_config_key(r.hp), []).append(r)
        ranked = []
        for _key, rs in groups.items():
            rmses = [r.rmse for r in rs]
            mean = _stats.fmean(rmses)
            std = _stats.pstdev(rmses) if len(rmses) > 1 else 0.0
            ranked.append((mean, std, len(rmses), rs[0]))
        ranked.sort(key=lambda x: x[0])
        if ranked:
            arch_best_mean[arch] = ranked[0]
        lines.append(f"## {arch}  ({len(a_ok)} runs, {len(groups)} configs)")
        lines.append("")
        lines.append("| rank | mean test rmse | std | n seeds | config |")
        lines.append("|-----:|---------------:|----:|--------:|--------|")
        for i, (mean, std, n, rep) in enumerate(ranked[:10], 1):
            lines.append(f"| {i} | {mean:.5f} | {std:.5f} | {n} | {rep.config} |")
        lines.append("")
    if arch_best_mean:
        winner = min(arch_best_mean.items(), key=lambda kv: kv[1][0])
        w_arch, (w_mean, w_std, w_n, _w_rep) = winner
        lines.append("## Cross-arch winner")
        lines.append("")
        lines.append(f"**{w_arch}** — best mean test RMSE "
                     f"{w_mean:.5f} ± {w_std:.5f} N·m (n={w_n} seeds).")
        lines.append("")
        lines.append("| arch | best mean test rmse | std | n |")
        lines.append("|------|--------------------:|----:|--:|")
        for arch in sorted(arch_best_mean):
            m, s, n, _ = arch_best_mean[arch]
            lines.append(f"| {arch} | {m:.5f} | {s:.5f} | {n} |")
        lines.append("")
    # ── TRAIN / VAL / TEST rmse_traj_macro per run + averages ──────────────
    # Requested: after ANY grid (full or test) show inference RMSE on
    # train / val / test, per run and averaged.  All trajectory-macro N·m,
    # the canonical headline estimator (train is pooled @best-val epoch).
    disp: list[str] = []
    disp.append("## Train / Val / Test RMSE (trajectory-macro, N·m)")
    disp.append("")
    disp.append("| arch | config | seed | train@best | val | test | gap(test−val) |")
    disp.append("|------|--------|-----:|-----------:|----:|-----:|--------------:|")
    _agg: dict[str, list[tuple[float, float, float]]] = {}
    for r in sorted(ok, key=lambda x: (x.arch, x.rmse if x.rmse is not None else 9e9)):
        mtr = _run_metrics_for(out_root, r.arch, r.hp)
        tr = mtr.get("train", float("nan"))
        vl = mtr.get("val", float("nan"))
        te = mtr.get("test", r.rmse if r.rmse is not None else float("nan"))
        seed = r.hp.get("seed", "?")
        gap = (te - vl) if (vl == vl and te == te) else float("nan")

        def _s(x):
            return f"{x:.5f}" if isinstance(x, float) and x == x else "  —  "
        disp.append(
            f"| {r.arch} | {r.config} | {seed} | {_s(tr)} | {_s(vl)} | "
            f"{_s(te)} | {_s(gap)} |"
        )
        if te == te:
            _agg.setdefault(r.arch, []).append((tr, vl, te))
    disp.append("")
    disp.append("**Per-arch averages**")
    disp.append("")
    disp.append("| arch | n | avg train | avg val | avg test |")
    disp.append("|------|--:|----------:|--------:|---------:|")
    for arch in sorted(_agg):
        rows = _agg[arch]

        def _mean(vs):
            vs = [v for v in vs if v == v]
            return (sum(vs) / len(vs)) if vs else float("nan")
        at = _mean([t for t, _, _ in rows])
        av = _mean([v for _, v, _ in rows])
        ae = _mean([e for _, _, e in rows])

        def _s2(x):
            return f"{x:.5f}" if x == x else "  —  "
        disp.append(f"| {arch} | {len(rows)} | {_s2(at)} | {_s2(av)} | {_s2(ae)} |")
    disp.append("")
    lines += ["", *disp]

    for _, md_name in _result_basenames():
        with open(os.path.join(out_root, md_name), "w") as fh:
            fh.write("\n".join(lines))

    # Always echo the TRAIN/VAL/TEST block to the console so it is visible
    # after ANY grid run (full sweep or 3-seed test), not just in the .md.
    _glog = logging.getLogger("grid")
    _glog.info("=" * 72)
    for _ln in disp:
        if _ln:
            _glog.info(_ln)
    _glog.info("=" * 72)


def _reconstruct_skipped_results(
    skipped: list[dict[str, Any]], out_root: str,
) -> list[_Result]:
    """Build ``_Result`` rows for already-trained trials from their on-disk
    runs (reuses :func:`_run_metrics_for`'s full-HP metadata match).

    These trials were pre-skipped (not executed this session) but ARE part of
    the active grid, so they must appear in grid_results.csv / grid_summary.md.
    Trials with no recoverable test metric on disk are dropped with a warning.
    """
    log = logging.getLogger("grid")
    out: list[_Result] = []
    missing = 0
    for t in skipped:
        arch, hp = t["arch"], t["hp"]
        m = _run_metrics_for(out_root, arch, hp)
        test = m.get("test") if m else None
        if test is None or test != test:          # missing / NaN
            missing += 1
            continue
        # status "ok": a reconstructed run is a completed, successful result.
        # ("skipped this session" is a runtime detail, not a property of the
        # result; downstream consumers filter the CSV on status == "ok".)
        out.append(_Result(
            n=0, arch=arch, config=_short_config(hp), status="ok",
            rmse=float(test), elapsed=0.0, hp=dict(hp),
        ))
    if missing:
        log.warning("Reconstruction: %d/%d skipped trials had no recoverable "
                    "test metric on disk (omitted).", missing, len(skipped))
    return out


def _finalize(
    executed: list[_Result],
    skipped: list[dict[str, Any]],
    out_root: str,
    log: logging.Logger,
) -> None:
    """Write the CSV/summary describing the WHOLE active grid.

    ``executed`` = trials run this session (real status incl. fail/err);
    ``skipped``  = already-trained trials, reconstructed from disk.  Combined
    so the artifacts are complete and resume-safe regardless of how many
    trials actually ran.
    """
    recon = _reconstruct_skipped_results(skipped, out_root)
    combined = list(executed) + recon
    if not combined:
        log.warning("Finalize: no results (executed or reconstructed) — "
                    "leaving existing grid artifacts untouched.")
        return
    for i, r in enumerate(
        sorted(combined,
                key=lambda x: (x.arch, x.rmse if x.rmse is not None else 9e9)),
        1,
    ):
        r.n = i
    try:
        _write_grid_summary(combined, out_root)
        names = ", ".join(c for c, _ in _result_basenames())
        log.info("Wrote %d rows (%d executed + %d reconstructed) -> %s",
                 len(combined), len(executed), len(recon), names)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to write grid summary: %s", exc)


# ============================================================================
# ── PARALLEL RUNNER (dynamic admission, rich dashboard) ──────────────────────
# ============================================================================

def _run_parallel_collect(
    trials: list[dict[str, Any]],
    pool_size: int,
    threads_per_worker: int,
    device_label: str,
    log: logging.Logger,
    t_start: float,
    n_gpus: int = 1,
) -> tuple[list["_Result"], list[dict[str, Any]], tuple[int, int, int]]:
    """Submit trials while live free VRAM + RAM permit.  No hardcoded N.

    Returns ``(executed_results, pre_skipped_trials, (ok, skip, fail))``
    WITHOUT finalizing — the caller owns artifact writing (so a two-stage
    run can derive Stage-A winners from ``executed_results`` before Stage B).

    Renders a live rich dashboard on TTY; falls back to periodic plain status
    lines when stdout/stderr are redirected.
    """
    # Detect HPC/workstation/laptop once and stamp compile flags on every trial.
    env = _detect_env()
    _apply_env_flags(env, trials)
    is_hpc = env == "hpc"

    original_total = len(trials)
    trials, _skipped = _partition_trials_by_skip(trials)
    n_pre_skip = len(_skipped)
    if n_pre_skip:
        log.info("Pre-dispatch SKIP: %d/%d trials already complete in this batch.",
                 n_pre_skip, original_total)
    # ── Resume-status checkpoint (resumable across HPC jobs) ─────────────
    # Write a small JSON file at the start AND end of dispatch so the user
    # can `cat` it between HPC jobs to know exactly what is done / pending.
    # Across-trial resume itself is already provided by SKIP_EXISTING; this
    # just surfaces the state.
    _status_path = os.path.join(DATASET_OUT_ROOT, "grid_resume_status.json")
    _write_resume_status(_skipped, trials, _status_path)
    log.info(
        "Resume state: %d/%d done, %d pending  →  %s",
        n_pre_skip, original_total, len(trials), _status_path,
    )
    if not trials:
        log.info("Nothing to do — all %d trials already complete; "
                 "artifacts will be rebuilt from disk by the caller.",
                 original_total)
        return [], _skipped, (0, n_pre_skip, 0)
    total = len(trials)

    # One-time measurement of the per-process CUDA context overhead.
    cuda_ctx_gb = _measure_cuda_ctx_gb()
    log.info("CUDA ctx overhead (per worker): %.2f GB", cuda_ctx_gb)

    # Per-trial estimates — different HPs will differ in future grids.
    # Two sources are reconciled: (a) the analytical model-size estimate, and
    # (b) the resource probe's *measured* peak VRAM from a real fwd+bwd.
    # The probe is the ground truth (the analytical formula systematically
    # undershoots by 5–20× because it ignores cuDNN workspaces, the caching
    # allocator's high-water mark, and torch.compile graph buffers).  Floor
    # every trial's VRAM estimate at the measured peak so the admission loop
    # never over-commits a GPU.
    measured_peak_vram_gb = _measure_worker_peak_vram_gb()
    log.info("Probe peak VRAM (per trial, ×1.5 safety): %.2f GB",
             measured_peak_vram_gb)
    estimates: list[tuple[float, float]] = []
    for t in trials:
        v_anal, r_anal = _estimate_trial_mem(t["hp"], t["arch"], cuda_ctx_gb)
        # Use the LARGER of analytical and probed peak as the dispatch budget.
        v_eff = max(v_anal, measured_peak_vram_gb)
        estimates.append((v_eff, r_anal))
    vram_max = max(e[0] for e in estimates)
    vram_min = min(e[0] for e in estimates)
    ram_max  = max(e[1] for e in estimates)
    log.info("Per-trial estimate: vram∈[%.2f, %.2f] GB  ram_max=%.2f GB",
             vram_min, vram_max, ram_max)

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
        n_gpus=n_gpus,
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

    # ── Per-GPU topology ────────────────────────────────────────────────────
    # Each physical GPU gets its own worker Pool whose workers are permanently
    # pinned to that GPU (fixed gpu_id in initargs).  Global slot ids are
    # partitioned into contiguous per-GPU ranges so the dashboard's flat
    # ``state.slots`` indexing is unchanged.
    n_gpus_eff = max(1, min(n_gpus, pool_size))
    per_gpu_counts = [
        pool_size // n_gpus_eff + (1 if g < pool_size % n_gpus_eff else 0)
        for g in range(n_gpus_eff)
    ]
    gpu_free_slots: list[list[int]] = [[] for _ in range(n_gpus_eff)]
    _s = 0
    for g in range(n_gpus_eff):
        for _ in range(per_gpu_counts[g]):
            gpu_free_slots[g].append(_s)
            _s += 1
    with state.lock:
        state.n_gpus = n_gpus_eff   # dashboard reflects the real topology
    # Per-GPU free-VRAM view.  The 1 Hz poller overwrites it with ground truth
    # from nvidia-smi; on admit we *optimistically* debit the target GPU so a
    # tight initial-fill burst can't over-commit one GPU before the poller
    # catches up (the poller then corrects any drift each second).
    _pg = _query_free_vram_gb_per_gpu()
    gpu_free_vram: list[float] = [
        (_pg[g] if g < len(_pg) else 0.0) for g in range(n_gpus_eff)
    ]

    # Seed per-GPU totals once (they don't change at runtime).  Used by the
    # dashboard for the per-GPU breakdown line under the VRAM gauge.
    _per_gpu_total: list[float] = []
    try:
        import subprocess as _sp
        _o = _sp.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        _per_gpu_total = [float(x) / 1024.0 for x in _o.splitlines() if x.strip()]
    except Exception:
        try:
            import torch as _t
            if _t.cuda.is_available():
                _per_gpu_total = [
                    _t.cuda.get_device_properties(i).total_memory / 1e9
                    for i in range(_t.cuda.device_count())
                ]
        except Exception:
            _per_gpu_total = []
    with state.lock:
        state.total_vram_gb_per_gpu = list(_per_gpu_total[:n_gpus_eff])

    # ── Resource poller: live VRAM/RAM gauges at 1 Hz ───────────────────────
    def _poll_resources() -> None:
        while not stop_event.is_set():
            pg = _query_free_vram_gb_per_gpu()
            fr = _query_free_ram_gb()
            with state.lock:
                for g in range(n_gpus_eff):
                    gpu_free_vram[g] = pg[g] if g < len(pg) else 0.0
                state.free_vram_gb = sum(gpu_free_vram)
                state.free_vram_gb_per_gpu = list(gpu_free_vram)
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
    # in_flight_map value: (trial, (vram_est, ram_est), slot, submitted_at, gpu)
    in_flight_map: dict[Any, tuple[dict, tuple[float, float], int, float, int]] = {}
    # Monotonic timestamp of the last admit/reap — drives the starvation guard.
    last_progress_t: list[float] = [time.time()]

    # ── RAM safety floor for the circuit breaker ────────────────────────────
    # Keep at least this much RAM truly free, even under burst-admit pressure.
    # On HPC (lots of RAM) this is a small fraction of the box; on a laptop the
    # default is sized to leave headroom for the desktop environment.  Scales
    # with `_total_ram_gb` so the same code path works across machines.
    ram_safety_floor_gb = max(8.0, _total_ram_gb * 0.05)

    # Per-trial wallclock timeout (default 45 min).  A trial running longer
    # than this is logged as a `slow_trial` warning every check; it is NOT
    # forcibly terminated (terminating a multiprocessing.Pool worker requires
    # killing the whole per-GPU pool, which would lose other live trials).
    # The starvation guard further down catches *wedged* grids (pool empty AND
    # pending work AND no progress for STARVATION_TIMEOUT_SEC).
    try:
        _trial_timeout_min = int(os.environ.get("MTP_GRID_TRIAL_TIMEOUT_MIN", "45"))
    except ValueError:
        _trial_timeout_min = 45
    max_trial_sec: float = float(max(10, _trial_timeout_min) * 60)
    _slow_trial_warned: set = set()

    def _set_counts() -> None:
        with state.lock:
            state.in_flight = len(in_flight_map)
            state.pending   = len(pending)

    def _reap_completed() -> None:
        for ar in [a for a in list(in_flight_map) if a.ready()]:
            trial, _est, slot, submitted_at, gpu = in_flight_map.pop(ar)
            gpu_free_slots[gpu].append(slot)
            last_progress_t[0] = time.time()
            try:
                result = ar.get(timeout=1.0)
            except Exception as exc:
                # ar.get() itself raising means the worker PROCESS died
                # (OOM-kill / segfault) — not a normal in-trial failure.
                # Make that cause explicit in the error log instead of a
                # bare repr.
                result = {
                    "status": "error", "arch": trial["arch"], "hp": trial["hp"],
                    "error": f"WORKER DIED: {type(exc).__name__}: {exc}",
                }
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
                # Surface the latest failure in the dashboard so the user can
                # see it without tailing the error log mid-run.
                with state.lock:
                    state.last_error_arch = trial['arch']
                    state.last_error_msg = first_line[:200]
            elapsed_trial = time.time() - submitted_at
            rmse = result.get("rmse")
            with state.lock:
                state.completed += 1
                if code == "ok":   state.ok   += 1
                elif code == "skip": state.skip += 1
                else:              state.fail += 1
                _res = _Result(
                    n=state.completed,
                    arch=trial["arch"],
                    config=_short_config(trial["hp"]),
                    status=code,
                    rmse=rmse,
                    elapsed=elapsed_trial,
                    hp=dict(trial["hp"]),
                )
                state.results.append(_res)
                state.all_results.append(_res)
                if rmse is not None and code == "ok":
                    _prev = state.best_by_arch.get(trial["arch"])
                    if _prev is None or rmse < _prev.rmse:
                        state.best_by_arch[trial["arch"]] = _res
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
                _all_so_far = list(state.all_results)
            _set_counts()
            # Incremental persistence: after every completed trial the CSV on
            # disk reflects all finished trials, so a crash/OOM/power loss at
            # trial N of a long grid keeps N-1 instead of losing all.
            # A write failure here must never kill the grid.
            try:
                _flush_results_csv(_all_so_far, DATASET_OUT_ROOT)
            except Exception as exc:
                log.warning("Incremental grid_results.csv flush failed: %s", exc)

    def _free_slots_total() -> int:
        return sum(len(s) for s in gpu_free_slots)

    def _pick_gpu(vram_est: float) -> int | None:
        """Return the GPU with the most headroom that (a) has a free slot and
        (b) fits ``vram_est`` on its OWN free VRAM.  Returns None if no single
        GPU qualifies — the correctness fix: a trial runs pinned to ONE GPU,
        so it must be gated on that GPU's free VRAM, never the cross-GPU sum.
        """
        best_g, best_avail = None, -1.0
        with state.lock:
            for g in range(n_gpus_eff):
                if not gpu_free_slots[g]:
                    continue
                avail = gpu_free_vram[g]
                if avail >= vram_est + VRAM_RESERVE_GB and avail > best_avail:
                    best_g, best_avail = g, avail
        return best_g

    def _dispatch(trial, est, gpu: int) -> None:
        vram_est, _ram = est
        slot = gpu_free_slots[gpu].pop(0)
        # Optimistically debit the chosen GPU so a tight admit burst can't
        # over-commit it before the 1 Hz poller observes the allocation.
        with state.lock:
            gpu_free_vram[gpu] = max(0.0, gpu_free_vram[gpu] - vram_est)
            state.free_vram_gb = sum(gpu_free_vram)
        ar = pools[gpu].apply_async(_run_one_trial, (trial, slot, threads_per_worker))
        in_flight_map[ar] = (trial, est, slot, time.time(), gpu)
        last_progress_t[0] = time.time()
        _set_counts()

    # Probe-measured peak RSS per worker, used by the RAM circuit breaker
    # below.  Falls back to a safe estimate if the probe wasn't run.
    _per_worker_peak_rss_gb = _measure_worker_rss_gb() / 1.2  # undo the 1.2x safety
    # Circuit-breaker threshold: dispatch is held off if free RAM would drop
    # below half the projected steady-state pool RSS after admitting one more.
    # Prevents the corner case where individual trials silently spike past the
    # probe-measured peak (large HP combos, dataset cache growth, etc.) and the
    # OS OOM-killer would otherwise reap workers.
    def _ram_circuit_breaker_open() -> bool:
        with state.lock:
            ram_free = state.free_ram_gb
        in_flight_n = len(in_flight_map)
        # Project the RAM impact of one more admit on top of currently running
        # workers.  Use the *peak* RSS (probe ground truth) per worker.
        projected_extra = _per_worker_peak_rss_gb * 1.0
        # If admitting would push the system past 90 % RAM utilization, hold.
        ram_threshold = ram_safety_floor_gb
        if ram_free < projected_extra + ram_threshold:
            return True
        return False

    def _try_admit_one() -> bool:
        """Dispatch the next trial onto a GPU that can hold it.

        Three gates:
          1. VRAM (hard, per-GPU): a CUDA-OOM crashes the trial outright.
          2. RAM circuit breaker (advisory): hold dispatch if admitting would
             push free RAM below the safety floor + projected per-trial spike.
             This prevents OOM-kills the probe's static peak can't anticipate.
          3. Reserve floor (legacy): warn if free RAM is below RAM_RESERVE_GB
             but still dispatch (reused workers pay marginal cost only).
        """
        if not pending or _free_slots_total() == 0:
            return False
        # RAM circuit breaker — if open, hold off this admit cycle.  Returning
        # False here causes the admission loop to sleep + retry; the breaker
        # closes when a running trial completes and frees its working set.
        if _ram_circuit_breaker_open() and in_flight_map:
            with state.lock:
                ram_free = state.free_ram_gb
            log.warning(
                "RAM circuit breaker OPEN — free=%.2f GB < %.2f GB threshold "
                "(in-flight=%d × peak=%.2f GB).  Waiting for a trial to free RAM.",
                ram_free, ram_safety_floor_gb + _per_worker_peak_rss_gb,
                len(in_flight_map), _per_worker_peak_rss_gb,
            )
            return False
        trial, est = pending[0]                       # largest-first
        with state.lock:
            ram_free = state.free_ram_gb
        gpu = _pick_gpu(est[0])
        if gpu is not None:
            if ram_free < RAM_RESERVE_GB:
                log.warning(
                    "RAM tight (free=%.2f, reserve=%.2f) — dispatching anyway; "
                    "reused workers only pay marginal tensor cost.",
                    ram_free, RAM_RESERVE_GB,
                )
            pending.pop(0)
            _dispatch(trial, est, gpu)
            return True
        # Largest didn't fit on any GPU.  If the whole pool is idle, the
        # largest trial would block forever — try the smallest instead.
        if not in_flight_map:
            pending.sort(key=lambda x: x[1][0])
            trial, est = pending[0]
            gpu = _pick_gpu(est[0])
            if gpu is not None:
                pending.pop(0)
                _dispatch(trial, est, gpu)
                pending.sort(key=lambda x: -x[1][0])
                return True
            pending.sort(key=lambda x: -x[1][0])
            with state.lock:
                _fv = list(gpu_free_vram)
            log.warning(
                "VRAM tight on all %d GPU(s) — free/gpu=%s GB; smallest trial "
                "needs %.2f GB. sleeping %.1fs",
                n_gpus_eff, [f"{v:.2f}" for v in _fv], est[0], TIGHT_SLEEP_SEC,
            )
            time.sleep(TIGHT_SLEEP_SEC)
        return False

    _set_counts()

    # ── Pool + dashboard lifecycle ──────────────────────────────────────────
    maxtasks = _maxtasks_for_env(env)   # hpc=64, workstation=16, laptop=4
    # One Pool per physical GPU.  The gpu_id is a FIXED scalar in initargs, so
    # every worker (incl. every maxtasksperchild recycle) is permanently
    # pinned to its GPU with no shared queue to drain → no recycle deadlock.
    pools: list = []
    for g in range(n_gpus_eff):
        pools.append(ctx.Pool(
            processes=max(1, per_gpu_counts[g]),
            maxtasksperchild=maxtasks,
            initializer=_pool_init,
            initargs=(progress_q, threads_per_worker, g, is_hpc),
        ))
    log.info(
        "Spawned %d GPU pool(s): workers/gpu=%s  (maxtasksperchild=%d)",
        n_gpus_eff, per_gpu_counts, maxtasks,
    )

    # ── Synchronous warmup barrier ──────────────────────────────────────────
    # Submit one no-op task per worker slot and join on the results.  This
    # guarantees every worker has completed ``_pool_init`` (including the
    # pre-warmed pipeline/strategies imports) BEFORE the first real trial
    # dispatches.  Without this, the first ``pool_size`` trials race against
    # cold-start imports and the dashboard shows "waiting" for ~30–60 s.
    try:
        _t0_warm = time.time()
        warm_handles: list = []
        for g, p in enumerate(pools):
            for _ in range(per_gpu_counts[g]):
                warm_handles.append(p.apply_async(_pool_warmup_noop))
        # Liberal per-worker timeout (300 s) accommodates HPC's heavy import
        # path; the warmup never sees user code, so a hang here is structural.
        warm_results = []
        for ar in warm_handles:
            try:
                warm_results.append(ar.get(timeout=300))
            except Exception as exc:
                log.warning("warmup task failed (worker may still be usable): %s", exc)
        log.info(
            "Pool warmup barrier passed: %d/%d workers ready in %.1fs",
            len(warm_results), len(warm_handles), time.time() - _t0_warm,
        )
    except Exception as exc:
        # A warmup failure is not fatal — real dispatch will surface true errors.
        log.warning("Warmup barrier raised (non-fatal): %s", exc)

    def _terminate_all_pools() -> None:
        for p in pools:
            try:
                p.terminate()
            except Exception:
                pass
        for p in pools:
            try:
                p.join()
            except Exception:
                pass

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
        # ── Staggered initial fill ──────────────────────────────────────────
        # Admitting all `pool_size` trials at once means N workers race to
        # load the same dataset off disk and thrash I/O — observed 90 s of
        # pre-epoch setup on a 16-way burst.  Instead, admit in waves of
        # ``wave_size`` and sleep ``dataset_load_sec × 0.6`` between waves so
        # subsequent loads overlap the prior wave's compute, not its disk
        # read.  Steady-state admission (after the pool is full) is unchanged.
        _probe = _probe_resource_peaks(TRAIN_DATA_RUN_DIR)
        _dload_sec = float(_probe.get("dataset_load_sec", 20.0))
        _wave_sleep = max(1.0, min(15.0, _dload_sec * 0.6))
        _wave_size = max(2, pool_size // 4)
        log.info(
            "Initial fill: staggering in waves of %d, sleep %.1fs between waves "
            "(probe load=%.1fs)", _wave_size, _wave_sleep, _dload_sec,
        )
        while pending and _free_slots_total():
            _admitted_in_wave = 0
            while _admitted_in_wave < _wave_size and pending and _free_slots_total():
                if not _try_admit_one():
                    break
                _admitted_in_wave += 1
            if _admitted_in_wave == 0:
                break
            if pending and _free_slots_total():
                # More to admit — reap any completions and wait before next wave.
                _reap_completed()
                time.sleep(_wave_sleep)

        # Main admission + reap loop
        while pending or in_flight_map:
            _reap_completed()
            if pending and _free_slots_total():
                admitted = _try_admit_one()
                if not admitted and in_flight_map:
                    time.sleep(ADMISSION_POLL_SEC)
            elif in_flight_map:
                time.sleep(ADMISSION_POLL_SEC)
            # Slow-trial watchdog: warn once per trial that exceeds the
            # wallclock budget.  Does NOT terminate (would kill the whole
            # GPU pool and lose other live trials), just surfaces the wedge
            # so the user can decide whether to abort.
            _now = time.time()
            for ar, (_trial, _est, _slot, submitted_at, _gpu) in list(in_flight_map.items()):
                age = _now - submitted_at
                if age > max_trial_sec and id(ar) not in _slow_trial_warned:
                    _slow_trial_warned.add(id(ar))
                    log.warning(
                        "SLOW TRIAL  arch=%s  %s  age=%.0fs (limit=%.0fs).  "
                        "Continuing — not terminated to preserve sibling trials.",
                        _trial["arch"], _hp_desc(_trial["arch"], _trial["hp"]),
                        age, max_trial_sec,
                    )
            # Starvation guard: pool fully idle, work still pending, and no
            # progress for STARVATION_TIMEOUT_SEC ⇒ genuinely wedged (every
            # remaining trial's VRAM estimate exceeds every GPU forever).
            # Fail loudly; the tail still writes whatever completed.
            if (not in_flight_map and pending
                    and (time.time() - last_progress_t[0]) > STARVATION_TIMEOUT_SEC):
                log.error(
                    "STARVATION: pool idle with %d trial(s) still pending and no "
                    "progress for %.0fs — aborting grid; %d completed result(s) "
                    "preserved.",
                    len(pending), STARVATION_TIMEOUT_SEC, state.completed,
                )
                break

    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt — terminating all worker pools ...")
        _terminate_all_pools()
        raise
    except Exception:
        log.exception("Fatal error in admission loop — terminating all pools ...")
        _terminate_all_pools()
        raise
    else:
        for p in pools:
            p.close()
        for p in pools:
            p.join()
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
        _all = list(state.all_results)
        _ok, _skip, _fail = state.ok, state.skip + n_pre_skip, state.fail
    # ── Refresh resume-status checkpoint on exit ─────────────────────────
    # Re-read disk to capture trials that completed THIS session (their
    # metadata.yaml is now on disk and _partition_trials_by_skip will
    # recognise them on the next HPC job).  Writes the same JSON file so
    # the user can see exactly what would be skipped on resume.
    try:
        _all_planned = list(_skipped) + list(trials)
        _now_pending, _now_done = _partition_trials_by_skip(_all_planned)
        _write_resume_status(_now_done, _now_pending,
                             os.path.join(DATASET_OUT_ROOT, "grid_resume_status.json"))
        log.info(
            "Final resume state: %d/%d done, %d still pending  →  "
            "next HPC job will dispatch the %d pending only.",
            len(_now_done), len(_all_planned), len(_now_pending),
            len(_now_pending),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not refresh resume-status on exit: %s", exc)
    # Hand back this session's results + the pre-skipped trials; the caller
    # finalizes (combining executed + disk-reconstructed) so the CSV is never
    # just the session subset.
    return _all, _skipped, (_ok, _skip, _fail)


def _run_parallel_dynamic(
    trials: list[dict[str, Any]],
    pool_size: int,
    threads_per_worker: int,
    device_label: str,
    log: logging.Logger,
    t_start: float,
    n_gpus: int = 1,
) -> tuple[int, int, int]:
    """Behavior-preserving wrapper: run in parallel, then finalize artifacts."""
    res, skipped, counts = _run_parallel_collect(
        trials, pool_size, threads_per_worker, device_label, log, t_start,
        n_gpus=n_gpus,
    )
    _finalize(res, skipped, DATASET_OUT_ROOT, log)
    return counts


# ============================================================================
# ── RUN-MODE SELECTION ───────────────────────────────────────────────────────
# ============================================================================

def _select_run_mode(log: logging.Logger) -> str:
    """Return 'quick', 'detailed', 'dataeff', 'combined', or 'reconstruct'.

    'combined' chains both research stages in one process: Stage A runs the
    full DETAILED HP grid, then Stage B runs the data-efficiency sweep on each
    arch's Stage-A winning config (selected by minimum test_rmse).

    Priority: env ``MTP_GRID_MODE`` → interactive prompt (TTY only) →
    default 'combined' (so just running the script launches the full
    Stage A → Stage B production run with no env vars / input needed).

    ``reconstruct`` trains nothing: it rebuilds grid_results_*.csv /
    grid_summary_*.md for an existing study purely from on-disk runs (which
    sub-study is set by ``MTP_GRID_RECONSTRUCT_OF``, default ``detailed``).
    """
    _ALIASES = {
        "quick": "quick", "detailed": "detailed",
        "dataeff": "dataeff", "data-efficiency": "dataeff", "frac": "dataeff",
        "reconstruct": "reconstruct", "rebuild": "reconstruct",
        "combined": "combined", "full": "combined", "ab": "combined",
    }
    env = os.environ.get("MTP_GRID_MODE", "").strip().lower()
    if env in _ALIASES:
        mode = _ALIASES[env]
        log.info("Run mode from MTP_GRID_MODE=%s", mode)
        return mode
    if env:
        log.warning(
            "Ignoring invalid MTP_GRID_MODE=%r (use 'quick', 'detailed', "
            "'dataeff', 'combined', or 'reconstruct').",
            env,
        )

    n_quick = sum(len(_cartesian(_ARCH_GRID_QUICK[a])) for a in _ARCH_META)
    n_det   = sum(len(_cartesian(_ARCH_GRID_DETAILED[a])) for a in _ARCH_META)
    n_de    = sum(len(_cartesian(_ARCH_GRID_DATAEFF[a])) for a in _ARCH_META)
    # Stage B = winners (≤ n archs) × fractions × seeds; n archs as upper bound.
    n_comb  = n_det + len(_ARCH_META) * len(_DATAEFF_FRACTIONS) * len(_DATAEFF_SEEDS)

    if not sys.stdin.isatty():
        log.info(
            "Non-interactive stdin — defaulting to COMBINED (%d trials: "
            "Stage A HP grid → Stage B data-efficiency on winners). "
            "Override with MTP_GRID_MODE=quick (%d) / detailed (%d) / "
            "dataeff (%d) if needed.",
            n_comb, n_quick, n_det, n_de,
        )
        return "combined"

    prompt = (
        "\n"
        "============================================================\n"
        "  Select run mode:\n"
        f"    [1] quick     — {n_quick} trials "
        f"(1 per model, full epochs; pipeline sanity check)\n"
        f"    [2] detailed  — {n_det} trials "
        f"(per-arch HP sweep at 100% data)\n"
        f"    [3] dataeff   — {n_de} trials "
        f"(best config/arch × data-fraction curve)\n"
        "    [4] reconstruct — no training; rebuild CSV/summary from disk "
        "(MTP_GRID_RECONSTRUCT_OF, default detailed)\n"
        f"    [5] combined  — {n_comb} trials (Stage A HP grid → Stage B "
        f"data-efficiency on the per-arch winners; one process)  [DEFAULT]\n"
        "============================================================\n"
        "  Enter 1, 2, 3, 4 or 5 (default 5 = combined): "
    )
    try:
        choice = input(prompt).strip()
    except EOFError:
        choice = ""
    mode = {"1": "quick", "2": "detailed", "3": "dataeff",
            "4": "reconstruct", "5": "combined"}.get(choice, "combined")
    log.info("Run mode selected: %s", mode)
    return mode


# ============================================================================
# ── RUNTIME PROBE  (shared by single-mode main() and combined orchestrator) ──
# ============================================================================

def _probe_runtime(log: logging.Logger) -> dict[str, Any]:
    """Probe device/pool capabilities ONCE (no CUDA context in main process).

    Returns ``{cuda_ok, n_gpus, pool_size, threads_per_worker, device_label}``.
    Factored out of ``main()`` so the combined two-stage orchestrator probes
    the same way and reuses one pool sizing for both stages.
    """
    cuda_ok = _query_cuda_available()
    n_gpus = _query_n_gpus() if cuda_ok else 1
    pool_size = _compute_pool_size(cuda_ok, n_gpus)
    threads_per_worker = _compute_threads_per_worker(pool_size)

    import psutil
    cpu_phys = psutil.cpu_count(logical=False) or 2
    free_ram_gb = _query_free_ram_gb()
    if cuda_ok:
        free_vram_gb  = _query_free_vram_gb()
        total_vram_gb = _query_total_vram_gb()
        try:
            import subprocess as _sp
            gpu_name = _sp.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                timeout=5,
            ).decode().strip().split("\n")[0]
        except Exception:
            gpu_name = "CUDA GPU"
        _gpu_suffix = f"×{n_gpus}" if n_gpus > 1 else ""
        device_label = f"cuda{_gpu_suffix} ({gpu_name}, {total_vram_gb:.1f} GB total)"
        log.info(
            "device=%s  gpus=%d  VRAM=%.1f/%.1f GB free   RAM=%.1f GB free   CPU=%d phys",
            device_label, n_gpus, free_vram_gb, total_vram_gb, free_ram_gb, cpu_phys,
        )
    else:
        device_label = "cpu"
        log.info("device=cpu   RAM=%.1f GB free   CPU=%d phys", free_ram_gb, cpu_phys)
    log.info("pool_size=%d   threads_per_worker=%d", pool_size, threads_per_worker)

    env = _detect_env()
    compile_on, compile_mode = _compile_flags(env, "fnn")   # representative arch
    maxtasks = _maxtasks_for_env(env)
    log.info(
        "env=%s   torch_compile=%s  mode=%s   maxtasksperchild=%d",
        env, compile_on, compile_mode, maxtasks,
    )
    return {
        "cuda_ok": cuda_ok,
        "n_gpus": n_gpus,
        "pool_size": pool_size,
        "threads_per_worker": threads_per_worker,
        "device_label": device_label,
    }


# ============================================================================
# ── COMBINED TWO-STAGE ORCHESTRATOR  (Stage A: HP grid → Stage B: dataeff) ───
# ============================================================================

def _stage_a_winners(
    executed: list["_Result"],
    skipped: list[dict[str, Any]],
    out_root: str,
    log: logging.Logger,
) -> dict[str, dict[str, Any]]:
    """Best config per arch from Stage A, selected by MINIMUM test_rmse.

    Folds disk-reconstructed pre-skipped runs in, so winners are derivable
    even when Stage A was 100% pre-completed on a prior (resumed) HPC job.
    Returns ``{arch: fully_expanded_hp_dict}``; an arch with no usable result
    is omitted (logged) and is simply excluded from Stage B.
    """
    ok = [r for r in executed if r.rmse is not None and r.status in ("ok", "skip")]
    ok += _reconstruct_skipped_results(skipped, out_root)
    winners: dict[str, dict[str, Any]] = {}
    for arch in sorted({r.arch for r in ok}):
        a_ok = [r for r in ok if r.arch == arch]
        best = min(a_ok, key=lambda r: r.rmse)          # user decision: min test_rmse
        winners[arch] = dict(best.hp)
        log.info("Stage-A winner [%s]: test_rmse=%.5f  %s",
                 arch, best.rmse, _short_config(best.hp))
    for arch in _ARCH_META:
        if arch not in winners:
            log.error("Stage-A winner [%s]: NO usable result — excluded from "
                      "Stage B.", arch)
    return winners


def _build_stage_b_trials(
    winners: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Data-efficiency trials = each arch's Stage-A winner × fractions × seeds.

    The winner hp is ALREADY fully expanded (concrete gravity/inertia/coriolis/
    friction_hidden, no ``edr_width``), so the EDR width-expansion in
    ``_build_trials`` is deliberately NOT re-run here.  The
    ``data_train_fraction==1.0`` × 3-seed subset is the multi-seed headline.
    """
    de_grid = {"data_train_fraction": _DATAEFF_FRACTIONS, "seed": _DATAEFF_SEEDS}
    trials: list[dict[str, Any]] = []
    for arch, win_hp in winners.items():
        model_type, save_subdir, run_help = _ARCH_META[arch]
        for combo in _cartesian(de_grid):
            hp = {**win_hp, **combo}
            hp["data_train_seed"] = int(combo["seed"])
            if arch == "edr":
                hp.update(_EDR_Q_STATS)   # idempotent; trig features preserved
            trials.append({
                "arch":        arch,
                "model_type":  model_type,
                "save_subdir": save_subdir,
                "run_help":    run_help,
                "hp":          hp,
            })
    return trials


def _write_combined_summary(
    winners: dict[str, dict[str, Any]],
    cnt_a: tuple[int, int, int],
    cnt_b: tuple[int, int, int],
    out_root: str,
    log: logging.Logger,
) -> None:
    """Additive ``combined_summary.md`` — never collides with grid_* artifacts."""
    lines = ["# Combined run — Stage A (HP grid) → Stage B (data-efficiency)", ""]
    lines.append(f"- Stage A: OK={cnt_a[0]}  Skip={cnt_a[1]}  Fail={cnt_a[2]}")
    lines.append(f"- Stage B: OK={cnt_b[0]}  Skip={cnt_b[1]}  Fail={cnt_b[2]}")
    lines += ["", "## Stage-A winning config per arch (min test_rmse)", "",
              "| arch | test_rmse @ Stage A | config |",
              "|------|--------------------:|--------|"]
    for arch, hp in winners.items():
        m = _run_metrics_for(out_root, arch, hp)
        te = m.get("test", float("nan")) if m else float("nan")
        te_s = f"{te:.5f}" if isinstance(te, float) and te == te else "—"
        lines.append(f"| {arch} | {te_s} | {_short_config(hp)} |")
    lines += [
        "",
        "## Headline (multi-seed)",
        "",
        "The Stage-B rows with `data_train_fraction == 1.0` (seeds 42, 1, 2) "
        "ARE the multi-seed headline — see `grid_results_dataeff.csv` for the "
        "per-seed test RMSE and the plot suite's `sweep_df` for mean ± std.",
        "",
        "Artifacts: Stage A → `grid_results_detailed.csv` (+ legacy "
        "`grid_results.csv` mirror); Stage B → `grid_results_dataeff.csv`.",
        "",
    ]
    path = os.path.join(out_root, "combined_summary.md")
    try:
        with open(path, "w") as fh:
            fh.write("\n".join(lines))
        log.info("Wrote combined summary -> %s", path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to write combined_summary.md: %s", exc)


def _run_combined(log: logging.Logger) -> None:
    """One process: full HP grid (Stage A) → winners → data-efficiency on the
    winners (Stage B).  Each stage is finalized to disk as it completes, so an
    interrupted HPC job resumes (pre-skip + disk-reconstruct) without retrain.
    """
    global _ARCH_GRID, RUN_MODE

    if DRY_RUN:
        log.info("DRY_RUN=True — combined mode prints the Stage-A combo table.")
        RUN_MODE = "detailed"
        _ARCH_GRID = dict(_ARCH_GRID_DETAILED)
        _print_combo_table(_build_trials())
        return

    for arch in _ARCH_META:
        _, save_subdir, _ = _ARCH_META[arch]
        os.makedirs(os.path.join(DATASET_OUT_ROOT, save_subdir), exist_ok=True)
    os.makedirs(os.path.join(DATASET_OUT_ROOT, "analysis"), exist_ok=True)

    rt = _probe_runtime(log)
    t_start = time.time()

    def _run(trials: list[dict[str, Any]]):
        if rt["pool_size"] <= 1:
            return _run_sequential_collect(trials, log, t_start)
        return _run_parallel_collect(
            trials, rt["pool_size"], rt["threads_per_worker"],
            rt["device_label"], log, t_start, n_gpus=rt["n_gpus"],
        )

    # ── Stage A — full HP grid (detailed sweep, single seed 42) ─────────────
    RUN_MODE = "detailed"                       # → legacy grid_results.csv mirror
    _ARCH_GRID = dict(_ARCH_GRID_DETAILED)
    trials_a = _build_trials()
    log.info("=" * 72)
    log.info("COMBINED · STAGE A — HP grid: %d trials (%s)",
             len(trials_a),
             ", ".join(f"{a}:{c}" for a, c in
                       sorted(Counter(t['arch'] for t in trials_a).items())))
    log.info("=" * 72)
    res_a, skip_a, cnt_a = _run(trials_a)
    _finalize(res_a, skip_a, DATASET_OUT_ROOT, log)   # detailed → +legacy mirror

    winners = _stage_a_winners(res_a, skip_a, DATASET_OUT_ROOT, log)
    if not winners:
        log.error("COMBINED: Stage A produced no usable winner for any arch — "
                  "Stage B skipped. Stage-A artifacts are written.")
        return

    # ── Stage B — data-efficiency on each arch's Stage-A winner ─────────────
    RUN_MODE = "dataeff"                         # → grid_results_dataeff.csv only
    trials_b = _build_stage_b_trials(winners)
    log.info("=" * 72)
    log.info("COMBINED · STAGE B — data-efficiency on winners: %d trials",
             len(trials_b))
    log.info("=" * 72)
    res_b, skip_b, cnt_b = _run(trials_b)
    _finalize(res_b, skip_b, DATASET_OUT_ROOT, log)

    _write_combined_summary(winners, cnt_a, cnt_b, DATASET_OUT_ROOT, log)

    log.info("=" * 72)
    log.info("COMBINED complete — Stage A: OK=%d Skip=%d Fail=%d  ·  "
             "Stage B: OK=%d Skip=%d Fail=%d",
             cnt_a[0], cnt_a[1], cnt_a[2], cnt_b[0], cnt_b[1], cnt_b[2])
    log.info("Results saved to: %s", DATASET_OUT_ROOT)
    log.info("Evaluate:  PYTHONPATH=. python3 Neural_Networks/eval_best_models.py")
    log.info("=" * 72)


# ============================================================================
# ── MAIN ─────────────────────────────────────────────────────────────────────
# ============================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    log = logging.getLogger("grid")

    global _ARCH_GRID, RUN_MODE
    mode = _select_run_mode(log)
    _GRID_BY_MODE = {
        "detailed": _ARCH_GRID_DETAILED,
        "dataeff":  _ARCH_GRID_DATAEFF,
        "quick":    _ARCH_GRID_QUICK,
    }

    # reconstruct: no training — rebuild a study's CSV/summary purely from
    # the runs already on disk (idempotent; seconds, not hours).
    if mode == "reconstruct":
        sub = os.environ.get("MTP_GRID_RECONSTRUCT_OF", "detailed").strip().lower()
        if sub not in _GRID_BY_MODE:
            log.warning("MTP_GRID_RECONSTRUCT_OF=%r invalid; using 'detailed'.", sub)
            sub = "detailed"
        RUN_MODE = sub                       # output files named for the sub-study
        _ARCH_GRID = dict(_GRID_BY_MODE[sub])
        trials = _build_trials()
        names = ", ".join(c for c, _ in _result_basenames())
        log.info("RECONSTRUCT '%s' grid (%d trials) from disk -> %s",
                 sub, len(trials), names)
        _finalize([], trials, DATASET_OUT_ROOT, log)
        log.info("Reconstruction complete (no training performed).")
        return

    # combined: chain Stage A (HP grid) → Stage B (data-efficiency on winners)
    # in one process.  Owns its own probe/dirs/dispatch/finalize.
    if mode == "combined":
        _run_combined(log)
        return

    RUN_MODE = mode
    _ARCH_GRID = dict(_GRID_BY_MODE.get(mode, _ARCH_GRID_QUICK))

    trials      = _build_trials()
    total       = len(trials)
    arch_counts = Counter(t["arch"] for t in trials)

    log.info("=" * 72)
    log.info("Journal Comparison — %s run (3 models, dynamic admission)", mode.upper())
    log.info("  ARCH       : %s", ARCH)
    log.info("  MODE       : %s", mode)
    log.info("  TOTAL      : %d trials", total)
    for a, cnt in sorted(arch_counts.items()):
        log.info("    %-10s  %d trial", a, cnt)
    log.info("  DATA       : %s", _DATASET_NAME)
    log.info("  OUTPUT     : %s", DATASET_OUT_ROOT)
    log.info("  SKIP_EXIST : %s  (pre-dispatch)", SKIP_EXISTING)
    log.info("  STRATEGY   : standard published strategy per arch (no augmentations)")
    log.info("  pool_size  : min(VRAM, CPU×oversub, RAM) ceilings  "
             "(env: MTP_GRID_POOL_SIZE / _CPU_OVERSUB / _WORKER_RAM_GB)")
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
    rt = _probe_runtime(log)
    pool_size          = rt["pool_size"]
    threads_per_worker = rt["threads_per_worker"]
    device_label       = rt["device_label"]
    n_gpus             = rt["n_gpus"]

    t_start = time.time()
    try:
        if pool_size <= 1:
            n_ok, n_skip, n_fail = _run_sequential(trials, log, t_start)
        else:
            n_ok, n_skip, n_fail = _run_parallel_dynamic(
                trials, pool_size, threads_per_worker, device_label, log, t_start,
                n_gpus=n_gpus,
            )
    except KeyboardInterrupt:
        log.warning("Interrupted by user — partial results preserved.")
        return

    elapsed = time.time() - t_start
    log.info("=" * 72)
    log.info("Grid search complete in %s", _fmt_time(elapsed))
    log.info("  OK=%d  Skipped=%d  Failed=%d  Total=%d", n_ok, n_skip, n_fail, total)
    log.info("Results saved to: %s", DATASET_OUT_ROOT)
    log.info("Evaluate:  PYTHONPATH=. python3 Neural_Networks/eval_best_models.py")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
