"""
Interactive Training CLI for Neural Network Torque Prediction.
train.py

This module is **not** imported by ``Neural_Networks.apps.grid_search`` — batch
grid sweeps call ``Neural_Networks.core.trainer.train_model`` directly with
``console=None``.  Interactive training here uses the same ``core`` primitives
(``train_epoch``, ``eval_epoch``, …) but keeps its own Rich live UI loop.

Usage
-----
    cd /home/san/Desktop/MTP_PINN
    python -m Neural_Networks.train

Workflow
--------
1. Model selection (menu 1–12; enter one id, a comma list e.g. ``1,4,5``, or a name)
2. Data preparation (filter, split, RNEA precomputation)
3. Hyperparameter configuration (rich docs, defaults by model type)
4. Training (early stopping, AMP, rich progress)
5. Post-training (metrics, plot, registry update)
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR,
    ExponentialLR, CyclicLR,
)
import yaml

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich import box
from rich.progress import (Progress, SpinnerColumn, BarColumn, TextColumn,
                           TimeElapsedColumn, MofNCompleteColumn, TimeRemainingColumn)
from rich.live import Live
from rich.prompt import Prompt, Confirm
from rich.markup import escape
from rich.console import Group

# Temporary Console() before the sub-package import below; overwritten immediately.
console = Console()

# Make sure project root is on path when running as module
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Neural_Networks.data.loader import make_dataloaders
from Neural_Networks.data.labels import (
    param_tag_from_metadata,
    pipeline_compact_codes,
    preprocessing_pipeline_table_rows,
    train_val_test_split_table_rows,
)
from Neural_Networks.core.collocation import CollocationSampler
from Neural_Networks.core.physics import ACTIVE_JOINTS
from Neural_Networks.models import (
    MODEL_REGISTRY, MODEL_CATEGORIES, MODEL_SAVE_DIRS,
    FNN_MODELS, PHYSICS_WEIGHT_MODELS,
    PHYSICS_INPUT_MODELS,
    LAGRANGIAN_MODELS,
    DECOMPOSED_MODELS, EQUATION_CONSTRAINED_MODELS,
)
# Rebind console to the shared tui singleton — ensures ONE Rich Console instance
# across train.py, tui.*, cli.*, so that Live/Progress contexts are consistent.
from Neural_Networks.tui.console import console as _tui_console, section, subsection
console = _tui_console
del _tui_console

# Architectures that expect analytical / RNEA torques in the preprocessed dataset.
_DATASET_NEEDS_ANALYTICAL_TORQUES: set[str] = (
    PHYSICS_WEIGHT_MODELS
    | PHYSICS_INPUT_MODELS
    | LAGRANGIAN_MODELS
    | EQUATION_CONSTRAINED_MODELS
    | DECOMPOSED_MODELS
)


def _models_need_rnea_in_dataset(model_types: str | list[str]) -> bool:
    if isinstance(model_types, str):
        mts = [model_types]
    else:
        mts = model_types
    return any(m in _DATASET_NEEDS_ANALYTICAL_TORQUES for m in mts)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

_APPS_DIR      = os.path.dirname(os.path.abspath(__file__))   # Neural_Networks/apps/
_NN_DIR        = os.path.dirname(_APPS_DIR)                    # Neural_Networks/
_PROJECT_ROOT  = os.path.dirname(_NN_DIR)                      # MTP_PINN/
TRAIN_DATA_DIR = os.path.join(_NN_DIR, "train_data")
MODELS_DIR     = os.path.join(_NN_DIR, "Trained_Models")
RAW_DIR        = os.path.join(_PROJECT_ROOT, "raw_samples")
XACRO_PATH     = os.path.join(_PROJECT_ROOT,
                               "robot_description", "urdf", "kikobot.xacro")
REGISTRY_FILE  = os.path.join(MODELS_DIR, "models_registry.yaml")


# =============================================================================
# HARDWARE AUTO-DETECTION
# =============================================================================

def detect_hardware() -> dict:
    """
    Detect GPU and system specs, return a training profile dict.

    Profiles:
      server  — A100 / H100 / A6000 (VRAM >= 40 GB)
      desktop — RTX 3080 / 4070 etc. (VRAM 8-39 GB)
      laptop  — RTX 3050 / 4050 etc. (VRAM < 8 GB)
      cpu     — no CUDA GPU

    Returns dict with profile, gpu_name, vram_gb, ram_gb, batch_size, etc.
    """
    import psutil

    ram_gb = psutil.virtual_memory().total / 1e9

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb  = props.total_memory / 1e9
        gpu_name = props.name
    else:
        vram_gb  = 0
        gpu_name = "CPU"

    if vram_gb >= 40:
        profile = "server"
    elif vram_gb >= 8:
        profile = "desktop"
    elif vram_gb > 0:
        profile = "laptop"
    else:
        profile = "cpu"

    hw_profiles = {
        # A100/H100/A6000 — large VRAM, many cores, fast PCIe
        "server": {
            "batch_size": 2048, "stride": 1, "epochs": 1000,
            "workers": 8, "prefetch": 8, "compile": True,
            "hidden_size": 256, "fc_layers": [256, 128],
        },
        # RTX 3080/4070/4090 etc. — ample VRAM
        "desktop": {
            "batch_size": 1024, "stride": 1, "epochs": 700,
            "workers": 6, "prefetch": 6, "compile": True,
            "hidden_size": 192, "fc_layers": [192, 96],
        },
        # RTX 3050/3060/4050 Laptop — limited VRAM (~4-8 GB)
        # stride=1 maximises training windows; batch=512 fits comfortably in 4 GB VRAM.
        # 500 epochs: PINN models need ≥200 epochs past the data-fitting plateau to diverge.
        "laptop": {
            "batch_size": 512, "stride": 1, "epochs": 500,
            "workers": 4, "prefetch": 4, "compile": True,
            "hidden_size": 128, "fc_layers": [128, 64],
        },
        # CPU-only — conservative settings
        "cpu": {
            "batch_size": 128, "stride": 5, "epochs": 200,
            "workers": 2, "prefetch": 2, "compile": False,
            "hidden_size": 64,  "fc_layers": [64],
        },
    }

    params = hw_profiles[profile]
    params.update({
        "profile":  profile,
        "gpu_name": gpu_name,
        "vram_gb":  round(vram_gb, 1),
        "ram_gb":   round(ram_gb, 1),
    })
    return params


# =============================================================================
# CLI HELPERS  (rich-styled)
# =============================================================================

def ask(prompt: str, default=None, cast=str, choices=None) -> Any:
    """
    Rich-styled prompt.  Press Enter to accept the default.
    """
    choices_hint = (
        "  [dim](" + "  ".join(
            f"[bold white]{c}[/bold white]" if str(c) == str(default) else str(c)
            for c in choices
        ) + ")[/dim]"
    ) if choices else ""

    display = f"[cyan]▸ {prompt}[/cyan]{choices_hint}"
    rich_default = str(default) if default is not None else ""

    while True:
        raw = Prompt.ask(display, console=console, default=rich_default).strip()

        if raw == "":
            if default is None:
                console.print("    [red]✗[/red] Required — please enter a value.")
                continue
            raw = str(default)

        try:
            val = cast(raw)
        except (ValueError, TypeError):
            console.print(f"    [red]✗[/red] Expected [yellow]{cast.__name__}[/yellow], "
                          f"got [red]'{escape(raw)}'[/red]")
            continue

        if choices:
            if str(raw).isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            raw_norm = str(raw).lower()
            val_norm = str(val).lower()
            valid_norms = [str(c).lower() for c in choices]
            if raw_norm not in valid_norms and val_norm not in valid_norms:
                opts = "  ".join(f"[bold white]{c}[/bold white]" if str(c).lower() == raw_norm
                                 else str(c) for c in choices)
                console.print(f"    [red]✗[/red] Choose: {opts}")
                continue

        return val


def ask_list(prompt: str, default: list, cast=int) -> list:
    """Ask for a comma-separated list; Enter accepts the default."""
    default_str = ",".join(str(v) for v in default)
    while True:
        raw = Prompt.ask(
            f"[cyan]▸ {prompt}[/cyan]",
            console=console, default=default_str
        ).strip()
        if not raw or raw == default_str:
            return default
        try:
            return [cast(v.strip()) for v in raw.split(",")]
        except ValueError:
            console.print(f"    [red]✗[/red] Comma-separated [yellow]{cast.__name__}[/yellow] "
                          f"values expected, e.g. [dim]{default_str}[/dim]")


# section() and subsection() are imported from Neural_Networks.tui.console above.

# =============================================================================
# HYPERPARAMETER MENUS (docs + grouping live in hp_registry)
# =============================================================================

from Neural_Networks.config.hp_registry import (
    COMMON_HP_DOCS,
    COMMON_STYLE_MODELS,
    GROUP_ORDER,
    HP_KEY_GROUPS,
    STRUCTURED_MODELS,
    apply_accurate_nominal_to_docs,
    apply_profile_to_hp_dict,
    activation_prompt_split_needed,
    dropout_prompt_split_needed,
    get_model_hp_docs,
    merge_doc_dicts_for_prompt,
    merge_shared_into_model_hp,
    model_needs_group,
    should_prompt_key,
    union_groups,
)

# Back-compat for any code expecting underscore-prefixed physics dict name
_PHYSICS_WEIGHT_HP = __import__(
    "Neural_Networks.config.hp_registry", fromlist=["PHYSICS_WEIGHT_HP"]
).PHYSICS_WEIGHT_HP


def print_hp_docs(hp_docs: dict):
    """Print hyperparameter documentation as a rich table."""
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold bright_white on grey23",
        border_style="dim cyan",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Parameter",   style="bold cyan",   no_wrap=True, min_width=18)
    table.add_column("Default",     style="bright_green",no_wrap=True, min_width=12)
    table.add_column("Options",     style="yellow",      no_wrap=False, min_width=14)
    table.add_column("Description + Effect", style="white", no_wrap=False)

    for name, info in hp_docs.items():
        default = str(info.get("default", "—"))
        choices = info.get("choices", None)
        opts    = ", ".join(choices) if choices else "—"
        body    = f"{info['desc']}\n[dim]{info['effect']}[/dim]"
        table.add_row(name, default, opts, body)

    console.print(table)


def _prompt_one_hp(name: str, info: dict) -> Any:
    default = info["default"]
    choices = info.get("choices", None)
    if isinstance(default, list):
        return ask_list(f"  {name}", default=default, cast=int)
    if isinstance(default, bool):
        return ask(
            f"  {name}", default=default,
            cast=lambda x: x.lower() in ("true", "1", "yes"),
            choices=["true", "false"],
        )
    if choices:
        return ask(f"  {name}", default=default, choices=choices)
    if isinstance(default, float):
        return ask(f"  {name}", default=default, cast=float)
    if isinstance(default, int):
        return ask(f"  {name}", default=default, cast=int)
    return ask(f"  {name}", default=default)


def _n_train_samples_from_dataset_meta(meta: dict | None) -> int:
    if not meta:
        return 0
    ss = (meta.get("split") or {}).get("stats") or {}
    return int(ss.get("train", {}).get("n_samples", 0) or 0)


def _inject_hw_into_doc_maps(
    common_map: dict[str, dict],
    specific_map: dict[str, dict[str, dict]],
    hw: dict,
) -> None:
    """Apply hardware defaults to copied doc dicts (mutates in place)."""
    hw_batch   = hw.get("batch_size")
    hw_hidden  = hw.get("hidden_size")
    hw_fc      = hw.get("fc_layers")
    hw_epochs  = hw.get("epochs")
    hw_stride  = hw.get("stride")
    if hw_batch is not None and "batch_size" in common_map:
        common_map["batch_size"]["default"] = hw_batch
    if hw_epochs is not None and "epochs" in common_map:
        common_map["epochs"]["default"] = hw_epochs
    for _m, spec in specific_map.items():
        if hw_hidden is not None and "hidden_size" in spec:
            spec["hidden_size"]["default"] = hw_hidden
        if hw_fc is not None and "fc_layers" in spec:
            spec["fc_layers"]["default"] = hw_fc
        if hw_stride is not None and "stride" in spec:
            spec["stride"]["default"] = hw_stride


def gather_hp(model_type: str, hw: dict | None = None) -> dict:
    """Interactively gather hyperparameters for the chosen model type.

    Parameters
    ----------
    model_type : str
        One of the 12 model type names.
    hw : dict | None
        Hardware profile from detect_hardware().  When provided, the profile's
        batch_size, hidden_size, fc_layers, epochs, and
        stride values override the static COMMON_HP_DOCS defaults so that
        pressing Enter for every prompt gives hardware-appropriate defaults.
    """
    specific, common = get_model_hp_docs(model_type)

    # --- Inject hardware defaults into the HP doc dicts ----------------------
    # We make shallow copies so we never mutate the module-level constants.
    if hw is not None:
        common = {k: dict(v) for k, v in common.items()}
        specific = {k: dict(v) for k, v in specific.items()}

        hw_batch   = hw.get("batch_size")
        hw_hidden  = hw.get("hidden_size")
        hw_fc      = hw.get("fc_layers")
        hw_epochs  = hw.get("epochs")
        hw_stride  = hw.get("stride")

        # common HP overrides
        if hw_batch  is not None and "batch_size"       in common:
            common["batch_size"]["default"]       = hw_batch
        if hw_epochs is not None and "epochs"           in common:
            common["epochs"]["default"]           = hw_epochs

        # model-specific HP overrides (present in specific but NOT common)
        if hw_hidden is not None and "hidden_size" in specific:
            specific["hidden_size"]["default"] = hw_hidden
        if hw_fc     is not None and "fc_layers"   in specific:
            specific["fc_layers"]["default"]   = hw_fc
        if hw_stride is not None and "stride"      in specific:
            specific["stride"]["default"]      = hw_stride

    all_docs = {**common, **specific}
    _ep = int(all_docs.get("epochs", {}).get("default", 500))
    apply_accurate_nominal_to_docs(
        model_type, all_docs, n_train_samples=None, epochs=_ep,
    )

    section("Hyperparameter Configuration")
    hw_tag = (f"  [dim]Hardware profile: [bold]{hw['profile'].upper()}[/bold] "
              f"({hw['gpu_name']}, {hw['vram_gb']} GB VRAM)[/dim]"
              if hw is not None else "")
    console.print(Panel(
        f"[bold]Model:[/bold] [bright_cyan]{model_type}[/bright_cyan]\n"
        "[dim]Press ENTER on any prompt to accept the default value shown in brackets.[/dim]"
        + ("\n" + hw_tag if hw_tag else ""),
        border_style="cyan", padding=(0, 2)
    ))
    print_hp_docs(all_docs)

    hp = {}
    subsection("Enter Hyperparameters")

    for name, info in specific.items():
        hp[name] = _prompt_one_hp(name, info)

    for name, info in common.items():
        hp[name] = _prompt_one_hp(name, info)

    hp["_n_train_samples"] = 0
    apply_profile_to_hp_dict(model_type, hp)
    return hp


def _queue_uses_key(model_types: list[str], key: str) -> bool:
    for m in model_types:
        sp, cm = get_model_hp_docs(m)
        if key in sp or key in cm:
            return True
    return False


def _ref_doc_for_key(model_types: list[str], key: str) -> tuple[str, dict[str, dict]]:
    """Pick a reference model that defines ``key`` and return merged doc map for defaults."""
    for m in model_types:
        sp, cm = get_model_hp_docs(m)
        merged = {**{k: dict(v) for k, v in cm.items()}, **{k: dict(v) for k, v in sp.items()}}
        if key in merged:
            return m, merged
    return model_types[0], {}


def gather_hp_for_models(
    model_types: list[str],
    hw: dict | None = None,
    dataset_meta: dict | None = None,
) -> dict[str, dict]:
    """
    Ask shared hyperparameters once for the union of model groups, then merge per model.
    Single-model queues delegate to ``gather_hp``.
    """
    if len(model_types) == 1:
        return {model_types[0]: gather_hp(model_types[0], hw=hw)}

    n_train = _n_train_samples_from_dataset_meta(dataset_meta)
    expert = (
        ask("Hyperparameter prompts", default="normal", choices=["normal", "expert"]).lower()
        == "expert"
    )

    per_spec: dict[str, dict[str, dict]] = {}
    per_common: dict[str, dict[str, dict]] = {}
    for m in model_types:
        sp, cm = get_model_hp_docs(m)
        per_spec[m] = {k: dict(v) for k, v in sp.items()}
        per_common[m] = {k: dict(v) for k, v in cm.items()}
    if hw is not None:
        for m in model_types:
            _inject_hw_into_doc_maps(per_common[m], {m: per_spec[m]}, hw)

    ugroups = union_groups(model_types)
    shared: dict[str, Any] = {}
    act_split = activation_prompt_split_needed(model_types)
    do_split = dropout_prompt_split_needed(model_types)

    section("Hyperparameter Configuration (batch)")
    console.print(Panel(
        f"[bold]Models:[/bold] [bright_cyan]{', '.join(model_types)}[/bright_cyan]\n"
        f"[dim]Detail: {'expert (all parameters)' if expert else 'normal (recommended subset)'}[/dim]",
        border_style="cyan", padding=(0, 2),
    ))

    def _silent_default(key: str) -> None:
        ref_m, merged = _ref_doc_for_key(model_types, key)
        if key not in merged:
            return
        ep = int(shared.get("epochs", merged.get("epochs", {}).get("default", 500)))
        frag = {key: dict(merged[key])}
        apply_accurate_nominal_to_docs(
            ref_m, frag, n_train_samples=n_train, epochs=ep,
        )
        shared[key] = frag[key]["default"]

    for group in GROUP_ORDER:
        if group not in ugroups:
            continue
        subsection(f"Shared — {group}")
        for key in HP_KEY_GROUPS.get(group, []):
            if not any(model_needs_group(m, group) for m in model_types):
                continue
            if not _queue_uses_key(model_types, key):
                continue
            if key == "activation" and act_split:
                continue
            if key == "dropout" and do_split:
                continue

            if not should_prompt_key(key, expert, shared):
                _silent_default(key)
                continue

            doc = merge_doc_dicts_for_prompt(model_types, key)
            ref_m, merged = _ref_doc_for_key(model_types, key)
            if key in merged:
                ep = int(shared.get("epochs", merged.get("epochs", {}).get("default", 500)))
                frag = {key: dict(merged[key])}
                apply_accurate_nominal_to_docs(
                    ref_m, frag, n_train_samples=n_train, epochs=ep,
                )
                doc = dict(frag[key])
            shared[key] = _prompt_one_hp(key, doc)

    if act_split:
        subsection("Shared — activation (split)")
        m_cb = next(m for m in model_types if m in COMMON_STYLE_MODELS)
        m_st = next(m for m in model_types if m in STRUCTURED_MODELS)
        _, ccb = get_model_hp_docs(m_cb)
        sst, _ = get_model_hp_docs(m_st)
        shared["activation_mlp"] = _prompt_one_hp(
            "activation (MLP / black-box-style)", dict(ccb["activation"]),
        )
        shared["activation_structured"] = _prompt_one_hp(
            "activation (Lagrangian / Decomposed)", dict(sst["activation"]),
        )

    if do_split:
        subsection("Shared — dropout (split)")
        m_cb = next(m for m in model_types if m in COMMON_STYLE_MODELS)
        m_st = next(m for m in model_types if m in STRUCTURED_MODELS)
        _, ccb = get_model_hp_docs(m_cb)
        sst, _ = get_model_hp_docs(m_st)
        shared["dropout_mlp"] = _prompt_one_hp(
            "dropout (MLP / black-box-style)", dict(ccb["dropout"]),
        )
        shared["dropout_structured"] = _prompt_one_hp(
            "dropout (Lagrangian / Decomposed)", dict(sst["dropout"]),
        )

    out: dict[str, dict] = {}
    for m in model_types:
        hp_m = merge_shared_into_model_hp(m, shared)
        hp_m["_n_train_samples"] = n_train
        apply_profile_to_hp_dict(m, hp_m)
        out[m] = hp_m
    return out


def get_default_hp(
    model_type: str,
    n_train_samples: int | None = None,
    epochs: int | None = None,
) -> dict:
    """Return default hyperparameters (accurate-nominal profile + per-model patches)."""
    specific, common = get_model_hp_docs(model_type)
    all_docs = {k: dict(v) for k, v in {**common, **specific}.items()}
    _ep = int(epochs) if epochs is not None else int(all_docs.get("epochs", {}).get("default", 500))
    apply_accurate_nominal_to_docs(
        model_type, all_docs, n_train_samples=n_train_samples, epochs=_ep,
    )
    hp = {name: info["default"] for name, info in all_docs.items()}
    hp["_n_train_samples"] = int(n_train_samples or 0)
    apply_profile_to_hp_dict(model_type, hp)
    return hp


# =============================================================================
# DATA PREPARATION STEP
# =============================================================================

def _load_run_metadata(run_path: str) -> dict | None:
    """Load run-level metadata.json (v3) or metadata.yaml (legacy)."""
    json_path = os.path.join(run_path, "metadata.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    yaml_path = os.path.join(run_path, "metadata.yaml")
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return None
    return None


def _print_dataset_summary(ds: dict, run_dir: str) -> None:
    """Print structured tables: preprocessing pipeline, then train/val/test counts (same filters for all splits)."""
    created = ds.get("created_at", "?")
    try:
        dt_obj = datetime.fromisoformat(str(created))
        created = dt_obj.strftime("%d %b %Y, %H:%M")
    except Exception:
        created = str(created)[:16]

    n_traj = ds.get("n_trajectories_processed", "?")

    try:
        ptag = param_tag_from_metadata(ds)
    except Exception:
        ptag = "?"

    header = (
        f"[bold]Run[/bold]        [bright_cyan]{escape(str(ds.get('run_name', os.path.basename(run_dir))))}[/bright_cyan]\n"
        f"[bold]Param tag[/bold]   [cyan]{escape(ptag)}[/cyan]\n"
        f"[bold]Created[/bold]    {escape(str(created))}\n"
        f"[bold]Source[/bold]     {escape(str(n_traj))} trajectories\n"
        f"[dim]{escape(run_dir)}[/dim]"
    )

    try:
        pipe_rows = preprocessing_pipeline_table_rows(ds)
    except Exception:
        pipe_rows = []

    pipe_table = Table(
        box=box.ROUNDED,
        border_style="green",
        header_style="bold white on grey23",
        title="[bold green]Preprocessing pipeline[/bold green] [dim](shared by train / val / test CSVs)[/dim]",
        padding=(0, 1),
        show_lines=False,
    )
    pipe_table.add_column("Quantity", style="bold cyan", min_width=22)
    pipe_table.add_column("Treatment", style="white", min_width=28)
    pipe_table.add_column("W", justify="right", style="dim", width=5)
    pipe_table.add_column("p", justify="right", style="dim", width=4)
    pipe_table.add_column("Mode / notes", style="dim white", min_width=18)

    if pipe_rows:
        for r in pipe_rows:
            pipe_table.add_row(
                escape(r["quantity"]),
                escape(r["treatment"]),
                escape(r["window"]),
                escape(r["poly"]),
                escape(r["mode_notes"]),
            )
    else:
        pipe_table.add_row("—", escape("(preprocessing metadata incomplete)"), "—", "—", "—")

    try:
        split_rows, trim_f = train_val_test_split_table_rows(ds)
    except Exception:
        split_rows, trim_f = [], {"front_pct": "?", "back_pct": "?", "total": "?"}

    split_table = Table(
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold white on grey23",
        title="[bold cyan]Sample counts by split[/bold cyan] [dim](after trim)[/dim]",
        padding=(0, 1),
    )
    split_table.add_column("Split", style="bold", justify="left", width=6)
    split_table.add_column("n samples", justify="right", style="bright_white", min_width=12)
    split_table.add_column("Config %", justify="right", style="dim", min_width=9)
    split_table.add_column("% of all rows", justify="right", style="dim", min_width=12)

    for r in split_rows:
        split_table.add_row(
            escape(r["split"]),
            escape(r["n_samples"]),
            escape(r["config_ratio"]),
            escape(r["fraction_of_all"]),
        )

    trim_note = (
        f"[dim]End trim: front {escape(trim_f.get('front_pct', '?'))}% · "
        f"back {escape(trim_f.get('back_pct', '?'))}%  ·  "
        f"rows summed: {escape(trim_f.get('total', '?'))}[/dim]"
    )

    body = Group(
        Text.from_markup(header),
        Text(""),
        pipe_table,
        Text(""),
        split_table,
        Text(""),
        Text.from_markup(trim_note),
    )
    console.print(Panel(
        body,
        title="[bold green]Dataset Selected[/bold green]",
        border_style="green",
        padding=(0, 2),
    ))


def _scan_existing_datasets() -> list[dict]:
    """Scan TRAIN_DATA_DIR for valid processed datasets (v3 metadata.json)."""
    if not os.path.isdir(TRAIN_DATA_DIR):
        return []

    valid = []
    for entry in sorted(os.scandir(TRAIN_DATA_DIR), key=lambda e: e.stat().st_mtime, reverse=True):
        if not entry.is_dir():
            continue
        meta = _load_run_metadata(entry.path)
        if meta is None:
            continue
        splits_ok = all(
            os.path.isdir(os.path.join(entry.path, s)) for s in ("train", "val", "test"))
        if not splits_ok:
            continue
        meta["run_name"] = entry.name
        meta["run_dir"]  = entry.path
        valid.append(meta)
    return valid


def _show_dataset_table(datasets: list[dict], model_types: str | list[str]) -> None:
    """Tabular picker: pipeline codes per signal + per-split sample counts; full detail after you select."""
    needs_physics = _models_need_rnea_in_dataset(model_types)

    t = Table(
        box=box.ROUNDED, border_style="bright_cyan",
        header_style="bold bright_white on grey23", padding=(0, 1),
        title="[bold bright_cyan]Available processed datasets[/bold bright_cyan]\n"
              "[dim]q / qd / qdd / τm = SG window·poly·mode (raw | lock→qd | L* = legacy); "
              "τ_apf = analytical post-filter[/dim]",
        show_lines=True,
    )
    t.add_column("#", style="bold white", justify="right", width=3)
    t.add_column("Run / param tag", style="cyan", justify="left", min_width=26, max_width=36)
    t.add_column("q", style="white", justify="center", width=10)
    t.add_column("qd", style="white", justify="center", width=14)
    t.add_column("qdd", style="white", justify="center", width=12)
    t.add_column("τm", style="white", justify="center", width=10)
    t.add_column("τ_apf", style="dim", justify="center", width=10)
    t.add_column("train", style="dim", justify="right", width=9)
    t.add_column("val", style="dim", justify="right", width=9)
    t.add_column("test", style="dim", justify="right", width=9)
    t.add_column("RNEA", style="green", justify="center", width=5)

    for idx, ds in enumerate(datasets, start=1):
        try:
            ptag = param_tag_from_metadata(ds)
        except Exception:
            ptag = ds.get("run_name", "?")
        try:
            codes = pipeline_compact_codes(ds)
        except Exception:
            codes = {"q": "?", "qd": "?", "qdd": "?", "tau_m": "?", "rnea": "?", "tau_apf": "?"}

        split_meta = ds.get("split", {}) or {}
        ss = split_meta.get("stats", {}) or {}

        def _ns(k: str) -> str:
            try:
                n = int(ss.get(k, {}).get("n_samples", 0) or 0)
                return f"{n:,}" if n else "0"
            except (TypeError, ValueError):
                return "?"

        pp = ds.get("preprocessing", {}) or {}
        has_rnea = bool(pp.get("tau_analytical", {}).get("rnea_enabled", False))
        rnea_str = "[green]Y[/green]" if has_rnea else "[dim]N[/dim]"
        if needs_physics and not has_rnea:
            rnea_str = "[bold red]![/bold red]"

        folder = escape(str(ds.get("run_name", "")))
        tag_cell = f"{escape(ptag)}\n[dim]{folder}[/dim]"

        t.add_row(
            str(idx),
            tag_cell,
            escape(codes["q"]),
            escape(codes["qd"]),
            escape(codes["qdd"]),
            escape(codes["tau_m"]),
            escape(codes["tau_apf"]),
            escape(_ns("train")),
            escape(_ns("val")),
            escape(_ns("test")),
            rnea_str,
        )

    console.print()
    console.print(t)
    console.print(
        "  [dim]# = index · After selection, full preprocessing + split tables are shown. "
        "RNEA [red]![/red] = needs physics but dataset has no RNEA.[/dim]"
    )
    console.print()


def select_existing_dataset(model_types: str | list[str]) -> tuple[str, dict] | None:
    """Show all available preprocessed datasets and let the user pick one."""
    datasets = _scan_existing_datasets()
    if not datasets:
        return None

    _mts = [model_types] if isinstance(model_types, str) else list(model_types)

    _show_dataset_table(datasets, _mts)
    console.print("  [dim]Enter [bold]0[/bold] to preprocess new data instead.[/dim]")
    while True:
        raw = Prompt.ask(
            f"[cyan]Select dataset [0-{len(datasets)}][/cyan]",
            console=console, default="1",
        ).strip()
        if raw.isdigit() and 0 <= int(raw) <= len(datasets):
            choice = int(raw)
            break
        console.print(f"    [red]Invalid.[/red] Enter 0-{len(datasets)}")

    if choice == 0:
        return None

    ds      = datasets[choice - 1]
    run_dir = ds["run_dir"]

    _print_dataset_summary(ds, run_dir)

    pp = ds.get("preprocessing", {}) or {}
    has_rnea = bool(pp.get("tau_analytical", {}).get("rnea_enabled", False))
    if any(m in DECOMPOSED_MODELS for m in _mts) and not has_rnea:
        console.print(
            "[bold yellow]Note:[/bold yellow] [dim]This dataset has RNEA disabled. "
            "DecomposedStructuredPINN will train without RNEA gravity prior.[/dim]"
        )

    return run_dir, ds


def data_preparation_step(model_types: str | list[str]) -> tuple[str, dict]:
    """
    Select a pre-built dataset or launch the GUI preprocessor.
    ``model_types`` may be one model or a list; dataset hints use the union of requirements.
    Returns (run_dir, meta).
    """
    section("Data Preparation")

    existing = _scan_existing_datasets()
    if existing:
        _mt_summary = (
            model_types
            if isinstance(model_types, str)
            else ", ".join(model_types)
        )
        console.print(Panel(
            f"  [bold]{len(existing)} preprocessed dataset(s)[/bold] found in "
            f"[dim]{TRAIN_DATA_DIR}[/dim]\n"
            f"  Planned model(s): [cyan]{_mt_summary}[/cyan]\n"
            f"  Select one or create a new dataset via the GUI preprocessor.",
            title="[bold bright_cyan]Dataset Selection[/bold bright_cyan]",
            border_style="bright_cyan", padding=(0, 2),
        ))
        use_existing = ask("Load existing dataset?", default="yes", choices=["yes", "no"])
        if use_existing.lower() in ("yes", "y"):
            result = select_existing_dataset(model_types)
            if result is not None:
                return result
        console.print()

    console.print(Panel(
        "No dataset selected.\n\n"
        "To create a new dataset, run the GUI preprocessor:\n"
        "  [bold cyan]python -m Neural_Networks.apps.preprocess[/bold cyan]\n\n"
        "The preprocessor will:\n"
        "  1. Load raw trajectories from raw_samples/\n"
        "  2. Let you configure all SG filter params per quantity\n"
        "  3. Save 10 CSV files + metadata.json per split\n"
        "  4. Output to Neural_Networks/train_data/run_<date>_<param_tag>_<id>/\n\n"
        "After building, re-run this training script to select the new dataset.",
        title="[bold yellow]Launch Preprocessor[/bold yellow]",
        border_style="yellow", padding=(0, 2),
    ))
    sys.exit(0)




# =============================================================================
# TRAINING INFRASTRUCTURE
# NOTE: build_model / build_optimizer / build_scheduler / pooled_rmse_numpy /
# trajectory_mean_rmse_numpy defined below are backward-compat stubs.
# They are OVERWRITTEN at the bottom of this file by canonical imports from
# Neural_Networks.core.builder and Neural_Networks.core.metrics.
# DO NOT edit these stubs — edit the sub-package modules instead.
# =============================================================================

def build_model(model_type: str, hp: dict, device: torch.device) -> nn.Module:
    """Backward-compat stub — overwritten by Neural_Networks.core.builder.build_model."""
    ModelClass = MODEL_REGISTRY[model_type]

    n_joints = 5

    if model_type == "BlackBoxFNN" or model_type == "PhysicsRegularizedFNN":
        kwargs = {
            "n_joints":      n_joints,
            "hidden_layers": hp.get("hidden_layers", [256, 256, 128]),
            "dropout":       hp.get("dropout", 0.1),
            "activation":    hp.get("activation", "silu"),
        }
        if model_type == "PhysicsRegularizedFNN":
            pass  # no extra constructor args
    elif model_type == "ResidualCorrectionFNN":
        kwargs = {
            "n_joints":      n_joints,
            "hidden_layers": hp.get("hidden_layers", [256, 256, 128]),
            "dropout":       hp.get("dropout", 0.1),
            "activation":    hp.get("activation", "tanh"),
        }
    elif model_type == "EquationConstrainedPINNFNN":
        kwargs = {
            "n_joints":      n_joints,
            "hidden_layers": hp.get("hidden_layers", [256, 256, 128]),
            "dropout":       hp.get("dropout", 0.1),
            "activation":    hp.get("activation", "silu"),
        }
    elif model_type == "LagrangianStructuredFNN":
        kwargs = {
            "n_joints":        n_joints,
            "inertia_layers":  hp.get("inertia_layers", [256, 512, 256]),
            "coriolis_layers": hp.get("coriolis_layers", [256, 512, 256]),
            "gravity_layers":  hp.get("gravity_layers", [256, 512, 256]),
            "friction_layers": hp.get("friction_layers", [128, 128]),
            "dropout":         hp.get("dropout", 0.1),
            "activation":      hp.get("activation", "tanh"),
        }
    elif model_type == "DecomposedStructuredPINNFNN":
        kwargs = {
            "n_joints":        n_joints,
            "inertia_layers":  hp.get("inertia_layers", [256, 512, 256]),
            "coriolis_layers": hp.get("coriolis_layers", [256, 512, 256]),
            "gravity_layers":  hp.get("gravity_layers", [256, 512, 256]),
            "friction_layers": hp.get("friction_layers", [128, 128]),
            "dropout":         hp.get("dropout", 0.1),
            "activation":      hp.get("activation", "tanh"),
            "lambda_data":     1.0,
            "lambda_spd":      hp.get("spd_weight", 0.01),
            "lambda_friction": hp.get("friction_weight", 0.01),
            "lambda_correction_reg":      hp.get("correction_reg_weight",      0.001),
            "lambda_nominal_consistency": hp.get("nominal_consistency_weight", 0.01),
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return ModelClass(**kwargs).to(device)


def build_optimizer(model: nn.Module, hp: dict) -> torch.optim.Optimizer:
    opt_name = hp.get("optimizer", "adamw").lower()
    lr       = hp.get("learning_rate", 3e-4)
    wd       = hp.get("weight_decay",  2e-3)
    phi_ratio = float(hp.get("phi_lr_ratio", 0.1))

    calib_params: list[nn.Parameter] = []
    main_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("tau_calib."):
            calib_params.append(p)
        else:
            main_params.append(p)

    if calib_params:
        param_groups = [
            {"params": main_params, "lr": lr, "weight_decay": wd},
            {"params": calib_params, "lr": lr * phi_ratio, "weight_decay": 0.0},
        ]
    else:
        param_groups = [{"params": list(model.parameters()), "lr": lr, "weight_decay": wd}]

    opts = {
        "adam":    lambda: Adam(param_groups),
        "adamw":   lambda: AdamW(param_groups),
        "sgd":     lambda: SGD(param_groups, momentum=0.9),
        "rmsprop": lambda: RMSprop(param_groups),
    }
    return opts.get(opt_name, opts["adamw"])()


def build_scheduler(optimizer, hp: dict, n_train_batches: int):
    sched_name = hp.get("lr_scheduler", "reduce_on_plateau").lower()
    epochs     = hp.get("epochs", 1000)
    lr         = hp.get("learning_rate", 3e-4)

    if sched_name == "none":
        return None
    if sched_name == "warmup_cosine":
        warmup_ep  = max(1, epochs // 20)
        min_factor = 0.01

        def _warmup_cosine_lambda(ep):
            if ep < warmup_ep:
                return 0.1 + 0.9 * (ep / warmup_ep)
            progress = (ep - warmup_ep) / max(1, epochs - warmup_ep)
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_cosine_lambda)
    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    if sched_name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 5), gamma=0.5)
    if sched_name == "reduce_on_plateau":
        # patience=25: enough buffer so physics-term noise doesn't prematurely halve LR.
        # factor=0.5: moderate reduction. threshold=1e-4: only react to meaningful changes.
        return ReduceLROnPlateau(optimizer, patience=25, factor=0.5,
                                 min_lr=lr * 0.01, threshold=1e-4)
    if sched_name == "onecycle":
        return OneCycleLR(optimizer, max_lr=lr * 10,
                          total_steps=epochs * n_train_batches)
    if sched_name == "exponential":
        return ExponentialLR(optimizer, gamma=0.95)
    if sched_name == "cyclic":
        return CyclicLR(optimizer, base_lr=lr * 0.1, max_lr=lr,
                        step_size_up=n_train_batches * 5, mode="triangular2")
    return None


def pooled_rmse_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """Unweighted RMSE over all elements (single pooled scalar)."""
    d = pred.astype(np.float64, copy=False) - target.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(d * d)))


def trajectory_mean_rmse_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    trajectories: list[dict],
) -> float:
    """Mean of per-trajectory RMSE — the canonical val_rmse metric.

    Each trajectory is evaluated independently and the results are averaged
    (macro-average).  This is unaffected by trajectory-length imbalance, so a
    short trajectory with high error contributes equally to a long easy one.

    Falls back to pooled RMSE when trajectory boundaries are unavailable.
    """
    if not trajectories:
        return pooled_rmse_numpy(pred, target)
    n = len(pred)
    per_traj: list[float] = []
    for traj in trajectories:
        s, e = traj["start_idx"], traj["end_idx_exclusive"]
        if e <= s or s >= n:
            continue
        e = min(e, n)
        per_traj.append(pooled_rmse_numpy(pred[s:e], target[s:e]))
    return float(np.mean(per_traj)) if per_traj else pooled_rmse_numpy(pred, target)


def _pearson_r_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute per-joint and summary metrics (physical N·m scale).

    * ``rmse`` / ``r2`` / … — per joint (axis 0).
    * ``rmse_mean`` — macro average of per-joint RMSEs (mean of √(MSE_j)).
    * ``rmse_pooled`` — √(mean squared error over all N×J elements); matches
      unweighted validation RMSE used for early stopping.
    * ``r2_mean`` — mean of per-joint R².
    * ``r2_overall`` — single R² on flattened predictions (one global SS_res/SS_tot).
    * Per-joint NRMSE uses (max−min) of target per joint as scale.
    """
    diff  = pred - target
    mse   = (diff ** 2).mean(axis=0)
    rmse  = np.sqrt(mse)
    mae   = np.abs(diff).mean(axis=0)
    max_e = np.abs(diff).max(axis=0)

    t_range = target.max(axis=0) - target.min(axis=0)
    nrmse   = rmse / (t_range + 1e-8)

    ss_res = (diff ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    r2     = 1.0 - ss_res / (ss_tot + 1e-10)

    pearson_r = np.array([
        _pearson_r_safe(pred[:, j], target[:, j])
        for j in range(pred.shape[1])
    ])
    exp_var = 1.0 - np.var(diff, axis=0) / (np.var(target, axis=0) + 1e-10)

    mse_pooled  = float(np.mean(diff.astype(np.float64, copy=False) ** 2))
    rmse_pooled = float(np.sqrt(mse_pooled))
    pv = pred.reshape(-1).astype(np.float64, copy=False)
    tv = target.reshape(-1).astype(np.float64, copy=False)
    ss_res_all = float(np.sum((pv - tv) ** 2))
    ss_tot_all = float(np.sum((tv - tv.mean()) ** 2))
    r2_overall = float(1.0 - ss_res_all / (ss_tot_all + 1e-10))

    return {
        "mse":               mse.tolist(),
        "rmse":              rmse.tolist(),
        "nrmse":             nrmse.tolist(),
        "mae":               mae.tolist(),
        "max_error":         max_e.tolist(),
        "r2":                r2.tolist(),
        "pearson_r":         pearson_r.tolist(),
        "explained_variance": exp_var.tolist(),
        "mse_mean":          float(mse.mean()),
        "rmse_mean":         float(rmse.mean()),
        "rmse_macro_mean":   float(rmse.mean()),
        "mse_pooled":        mse_pooled,
        "rmse_pooled":       rmse_pooled,
        "r2_overall":        r2_overall,
        "nrmse_mean":        float(nrmse.mean()),
        "mae_mean":          float(mae.mean()),
        "r2_mean":           float(r2.mean()),
        "pearson_r_mean":    float(np.mean(pearson_r)),
    }



# ─── DEAD CODE REMOVED ─────────────────────────────────────────────────────────
# Local copies of _forward_pass, _split_decomposed_physics, _compute_loss,
# train_epoch, eval_epoch, save_comparison_plot, save_architecture_summary,
# _fmt_time, _make_serializable, update_registry, and PhysicsWeightScheduler
# were defined here but are now imported from their canonical locations in
# Neural_Networks.core.trainer (see SUB-PACKAGE IMPORT REDIRECTIONS below).
# ────────────────────────────────────────────────────────────────────────────────


# ─── DEAD CODE REMOVED (PhysicsWeightScheduler, _compute_loss, train_epoch, ──
# ─── eval_epoch, save_comparison_plot, save_architecture_summary, _fmt_time, ──
# ─── _make_serializable, update_registry) — all imported from core.trainer. ───


# =============================================================================
# SUB-PACKAGE IMPORT REDIRECTIONS
# =============================================================================
# The functions defined earlier in this file (detect_hardware, ask, ask_list,
# build_model, etc.) are the original implementations.  The imports below
# redirect all of their names to the canonical sub-package versions.
# Python's module-level name resolution means the LAST assignment wins, so
# placing these imports AFTER the function definitions causes every caller in
# the rest of this file (and external importers) to use the sub-package code.
#
# Legacy bodies above are preserved only for backward-compatibility with any
# external code that does ``from Neural_Networks.train import detect_hardware``.
# =============================================================================

from Neural_Networks.config.hardware    import detect_hardware       # noqa: E402
from Neural_Networks.cli.prompts        import ask, ask_list         # noqa: E402
from Neural_Networks.tui.hp_display     import print_hp_docs         # noqa: E402
from Neural_Networks.cli.hp_wizard      import (                     # noqa: E402
    _n_train_samples_from_dataset_meta,
    _inject_hw_into_doc_maps,
    _prompt_one_hp,
    _queue_uses_key,
    _ref_doc_for_key,
    gather_hp,
    gather_hp_for_models,
    get_default_hp,
)
from Neural_Networks.data.scanner       import (                     # noqa: E402
    load_run_metadata as _load_run_metadata_pkg,
)
from Neural_Networks.tui.dataset_display import (                    # noqa: E402
    print_dataset_summary as _print_dataset_summary,
    show_dataset_table    as _show_dataset_table,
)
from Neural_Networks.core.builder       import (                     # noqa: E402
    build_model,
    build_optimizer,
    build_scheduler,
)
from Neural_Networks.core.metrics       import (                     # noqa: E402
    pooled_rmse_numpy,
    trajectory_mean_rmse_numpy,
    macro_rmse_numpy,
    _pearson_r_safe,
    compute_metrics,
)
from Neural_Networks.core.trainer       import (                     # noqa: E402
    _fmt_time,
    _make_serializable,
    PhysicsWeightScheduler,
    _forward_pass,
    train_epoch,
    eval_epoch,
    save_comparison_plot,
    save_architecture_summary,
    update_registry,
)
from Neural_Networks.cli.menus          import (                     # noqa: E402
    QUICK_TEST_SENTINEL as _QUICK_TEST_SENTINEL,
    data_preparation_step,
    _build_model_menu,
    _select_model_types,
    print_batch_results_table,
    _run_quick_test_all as _cli_run_quick_test_all,
)


def _run_quick_test_all(model_list: list, hw: dict | None = None) -> None:
    """Thin shim that forwards to the canonical cli.menus implementation.

    Keeps the (model_list, hw) call signature used by main() while the real
    logic lives in cli.menus — so any HP-profile / registry / metrics changes
    land in exactly one place.  The local ``train_model`` wrapper (defined
    later in this module) supplies the models_dir / registry_file / nn_dir
    kwargs that ``core.trainer.train_model`` now requires, so we pass it in
    explicitly instead of letting cli.menus fall back to the bare core one.
    """
    _cli_run_quick_test_all(
        model_list,
        TRAIN_DATA_DIR,
        hw=hw,
        train_model_fn=train_model,
    )

# ─── Local _scan_existing_datasets wrapper (uses module-level TRAIN_DATA_DIR) ─
# data.scanner.scan_existing_datasets() requires an explicit path argument;
# we keep a zero-arg wrapper here so internal callers are unchanged.
def _scan_existing_datasets() -> list[dict]:  # noqa: redefinition-of-function
    from Neural_Networks.data.scanner import scan_existing_datasets
    return scan_existing_datasets(TRAIN_DATA_DIR)


def _make_live_panel(progress, epoch_lines: list, epoch: int, epochs: int,
                     best_val: float,
                     gpu_mem_gb: float | None = None,
                     gpu_total_gb: float | None = None,
                     samples_per_sec: float | None = None,
                     current_lr: float | None = None,
                     grad_norm: float | None = None) -> Group:
    """Build the live display group: progress bar + last-10-epoch log + runtime stats."""
    panels: list = [progress]

    if epoch_lines:
        visible = epoch_lines[-10:]
        panels.append(Panel("\n".join(visible), border_style="dim", padding=(0, 1)))

    parts = []
    if gpu_mem_gb is not None and gpu_total_gb and gpu_total_gb > 0:
        pct = min(1.0, gpu_mem_gb / gpu_total_gb)
        bar_len = 12
        filled  = int(pct * bar_len)
        bar     = "█" * filled + "░" * (bar_len - filled)
        mem_col = "red" if pct > 0.85 else ("yellow" if pct > 0.65 else "green")
        parts.append(f"[bold]GPU[/bold] [{mem_col}]{bar} {gpu_mem_gb:.1f}/{gpu_total_gb:.0f}GB[/{mem_col}]")
    if samples_per_sec is not None:
        parts.append(f"[bold]Speed[/bold] [bright_cyan]{samples_per_sec:,.0f}[/bright_cyan][dim] smp/s[/dim]")
    if current_lr is not None:
        parts.append(f"[bold]LR[/bold] [yellow]{current_lr:.2e}[/yellow]")
    if grad_norm is not None:
        gnorm_col = "red" if grad_norm > 0.9 else ("yellow" if grad_norm > 0.5 else "bright_cyan")
        parts.append(f"[bold]‖∇‖[/bold] [{gnorm_col}]{grad_norm:.3f}[/{gnorm_col}]")
    parts.append(f"[bold]BestRMSE[/bold] [bright_green]{best_val:.5f}[/bright_green][dim] N·m[/dim]")
    if parts:
        panels.append(Text.from_markup("  " + "  │  ".join(parts)))

    return Group(*panels)


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train_model(run_dir: str, model_type: str, hp: dict):
    """Full training pipeline for one model."""
    section(f"Training  {model_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_tag = "[bold green]GPU ✓[/bold green]" if device.type == "cuda" else "[yellow]CPU[/yellow]"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark        = True
        torch.backends.cudnn.deterministic    = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.cuda.empty_cache()
        import logging as _logging
        for _lg in ("torch._inductor", "torch._dynamo", "torch.fx",
                    "torch._inductor.select_algorithm"):
            _logging.getLogger(_lg).setLevel(_logging.ERROR)
        _cache_dir = os.path.join(_NN_DIR, ".torch_compile_cache")
        os.makedirs(_cache_dir, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _cache_dir)

    needs_physics   = (model_type in PHYSICS_WEIGHT_MODELS or
                       model_type in PHYSICS_INPUT_MODELS)
    mode            = "pointwise"

    pin_memory = device.type == "cuda"
    import psutil
    _ncpu   = os.cpu_count() or 4
    _ram_gb = psutil.virtual_memory().total / 1e9
    _max_workers = 4 if _ram_gb < 20 else 8
    _nw_env = os.environ.get("NN_NUM_WORKERS", "").strip()
    if _nw_env.isdigit():
        num_workers = max(0, int(_nw_env))
    else:
        num_workers = max(2, min(_max_workers, _ncpu // 2)) if device.type == "cuda" else 0
    _pf_env = os.environ.get("NN_PREFETCH", "").strip()
    if _pf_env.isdigit():
        _prefetch = max(2, int(_pf_env))
    else:
        _prefetch = 6 if _ram_gb < 20 else 10

    loaders = make_dataloaders(
        run_dir             = run_dir,
        batch_size          = hp.get("batch_size", 2048),
        mode                = mode,
        seq_len             = hp.get("seq_len", 50),
        stride              = hp.get("stride", 1),
        normalise           = True,
        num_workers         = num_workers,
        pin_memory          = pin_memory,
        prefetch_factor     = _prefetch,
        drop_last           = True,
        data_train_fraction = float(hp.get("data_train_fraction", 1.0)),
        data_train_seed     = int(hp.get("data_train_seed",
                                          hp.get("_grid_seed", 0)) or 0),
    )

    model           = build_model(model_type, hp, device)
    _model_cls_name = model.__class__.__name__
    _model_n_params = model.count_parameters() if hasattr(model, "count_parameters") else 0

    # torch.compile: auto on ≥16 GB VRAM; opt-in via hp or NN_TORCH_COMPILE on smaller GPUs
    _compiled    = False
    _gpu_vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                    if device.type == "cuda" else 0)
    _compile_env = os.environ.get("NN_TORCH_COMPILE", "").strip().lower() in ("1", "true", "yes")
    _want_compile = (
        device.type == "cuda"
        and (
            _gpu_vram_gb >= 16
            or bool(hp.get("torch_compile", False))
            or _compile_env
        )
    )
    _compile_mode = str(hp.get("torch_compile_mode", "default")).strip().lower()
    if _compile_mode not in ("default", "reduce-overhead", "max-autotune"):
        _compile_mode = "default"
    if _want_compile:
        if _gpu_vram_gb < 16:
            console.print(
                "  [yellow]torch.compile[/yellow]  [dim]opt-in on low VRAM — "
                "first epoch may be slow while kernels compile.[/dim]"
            )
        else:
            console.print("  [dim]Compiling model (first run ~30 s, cached afterwards)…[/dim]")
        try:
            model     = torch.compile(model, mode=_compile_mode)
            _compiled = True
        except Exception:
            pass
    elif device.type == "cuda":
        console.print(
            f"  [dim]Skipping torch.compile (VRAM={_gpu_vram_gb:.0f} GB; "
            f"set hp torch_compile or NN_TORCH_COMPILE=1 to opt in)[/dim]"
        )

    optimizer = build_optimizer(model, hp)

    _colloc_sampler: CollocationSampler | None = None
    if (
        model_type in EQUATION_CONSTRAINED_MODELS
        and float(hp.get("lambda_collocation", 0.0)) > 0.0
    ):
        try:
            _colloc_sampler = CollocationSampler(loaders["train"].dataset)
            if not _colloc_sampler.ok:
                _colloc_sampler = None
        except Exception as _e:
            logger.warning("Collocation sampler init failed: %s", _e)
            _colloc_sampler = None

    n_train_batches = len(loaders["train"])

    onecycle_sched = None
    if hp.get("lr_scheduler", "reduce_on_plateau") == "onecycle":
        onecycle_sched = build_scheduler(optimizer, hp, n_train_batches)
        scheduler      = None
    else:
        scheduler = build_scheduler(optimizer, hp, n_train_batches)

    _phys_sched: PhysicsWeightScheduler | None = None
    if model_type in PHYSICS_WEIGHT_MODELS:
        _phys_sched = PhysicsWeightScheduler.from_hp(hp)
        # LossNormaliser is OPT-IN (not default).  Empirically, on this data-rich
        # regime (~500k samples of real dynamics with calibrated τ_nom bias),
        # forcing physics to unit-scale with data over-amplifies an imperfect
        # prior — the model gets dragged toward biased τ_calib(τ_nom) and
        # val_rmse degrades after only ~10 epochs.  Leaving losses at their
        # natural magnitudes lets data dominate the gradient while physics
        # acts as a gentle regulariser, which matches the best historical
        # runs (EC-PINN ep881 val_rmse=0.0938).  Enable only when the dataset
        # is genuinely small or τ_calib is known-accurate:
        #     hp["enable_loss_normaliser"] = True
        if bool(hp.get("enable_loss_normaliser", False)):
            from Neural_Networks.core.trainer import LossNormaliser
            _target = model._orig_mod if hasattr(model, "_orig_mod") else model
            _target._loss_normaliser = LossNormaliser(
                beta=float(hp.get("loss_norm_beta", 0.98)),
            )

    scaler       = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
    gpu_name     = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    gpu_total_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                    if device.type == "cuda" else 0.0)
    gpu_info     = f" [dim]({gpu_name})[/dim]" if gpu_name else ""
    amp_tag      = "[bold green]AMP ✓[/bold green]" if scaler else "[dim]—[/dim]"
    compile_tag  = "[bold green]compiled ✓[/bold green]" if _compiled else "[dim]eager[/dim]"

    console.print(Panel(
        f"[bold]Device:[/bold]      {device_tag}{gpu_info}   "
        f"[bold]Workers:[/bold] [dim]{num_workers}[/dim]\n"
        f"[bold]Model:[/bold]       [bright_cyan]{_model_cls_name}[/bright_cyan]   "
        f"[bold]Params:[/bold] [bright_green]{_model_n_params:,}[/bright_green]   "
        f"[bold]AMP:[/bold] {amp_tag}   [bold]compile:[/bold] {compile_tag}\n"
        f"[bold]Batch size:[/bold]  [yellow]{hp.get('batch_size', 2048)}[/yellow]   "
        f"[bold]Batches:[/bold] {n_train_batches}   "
        f"[bold]Optimizer:[/bold] [yellow]{hp.get('optimizer','adamw')}[/yellow]   "
        f"[bold]LR:[/bold] [yellow]{hp.get('learning_rate',3e-4)}[/yellow]   "
        f"[bold]Scheduler:[/bold] [yellow]{hp.get('lr_scheduler','reduce_on_plateau')}[/yellow]"
        + (
            f"\n[bold]Physics loss:[/bold]"
            f"  [magenta]{_phys_sched.describe()}[/magenta]"
            if (model_type in PHYSICS_WEIGHT_MODELS and _phys_sched is not None)
            else ""
        ),
        title=f"[bold bright_cyan]{model_type}[/bold bright_cyan]",
        border_style="cyan", padding=(0, 2)
    ))

    best_val_loss      = math.inf
    best_val_rmse      = math.inf   # best unweighted normalised RMSE (used for early stopping)
    best_val_rmse_phys = math.inf   # best val RMSE in physical units N·m (display only)
    best_epoch         = 0          # epoch at which best_val_rmse_phys was recorded
    best_state         = None
    patience_counter = 0
    patience         = hp.get("patience", 100)
    use_early_stop   = hp.get("early_stopping", True)
    epochs           = hp.get("epochs", 500)
    _min_delta       = float(hp.get("min_delta", 1e-4))
    _early_metric = str(hp.get("early_stop_metric", "val_rmse")).strip().lower()
    if _early_metric not in ("val_rmse", "val_loss"):
        _early_metric = "val_rmse"
    best_val_loss_track = math.inf  # for early_stop_metric == val_loss
    _best_ckpt_cache_dir = os.path.join(
        MODELS_DIR, "._train_cache", f"{model_type}_{uuid.uuid4().hex[:10]}",
    )
    history          = {
        "train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": [],
        "w_d": [], "w_p": [],
    }
    _grad_norm       = 0.0
    _epoch_lines: list[str] = []

    console.print()
    if _early_metric == "val_rmse":
        console.print(
            f"  [dim]Best val_rmse checkpoint cache:[/dim] [dim]{_best_ckpt_cache_dir}[/dim]"
        )
    # Val dataset normalisation stats (used for physical-unit RMSE during training)
    _tau_std_val  = loaders["val"].dataset.std_tau     # shape (n_joints,) float32
    _tau_mean_val = loaders["val"].dataset.mean_tau    # shape (n_joints,) float32
    # Trajectory boundaries for macro-average RMSE (one RMSE per trajectory, then averaged)
    _val_trajectories: list[dict] = (
        loaders["val"].dataset.metadata
        .get("split", {}).get("stats", {}).get("val", {}).get("trajectories", [])
    )

    t0 = time.time()

    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=34, style="dim cyan", complete_style="bright_green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]ETA[/dim]"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )
    task = progress.add_task(f"[cyan]Epoch 0/{epochs}", total=epochs)

    stopped_early    = False
    _samples_per_sec = None
    _gpu_mem_gb      = None
    _current_lr      = hp.get("learning_rate", 3e-4)
    _wd_eff          = 1.0
    _wp_eff          = 0.0

    if model_type in PHYSICS_WEIGHT_MODELS and patience < 80:
        console.print(
            "  [yellow]⚠[/yellow]  [dim]patience < 80 — fixed-α PINN models "
            "often need ≥80 epochs without improvement before training has "
            "actually converged.[/dim]"
        )

    with Live(_make_live_panel(progress, _epoch_lines, 0, epochs, math.inf),
              console=console, refresh_per_second=4, transient=True) as live:

        if _phys_sched is not None:
            _wd_eff, _wp_eff = 1.0, _phys_sched.w_p

        for epoch in range(1, epochs + 1):
            _ep_t0 = time.time()
            train_loss, _grad_norm, train_rmse_unw = train_epoch(
                model, loaders["train"], optimizer,
                device, model_type,
                _wd_eff, _wp_eff, hp, onecycle_sched, scaler,
                feature_noise_std=hp.get("feature_noise_std", 0.01),
            )
            _ep_dt = time.time() - _ep_t0
            _n_train         = len(loaders["train"].dataset)
            _samples_per_sec = _n_train / max(_ep_dt, 1e-6)

            # Optional physics collocation step (Equation-Constrained only)
            if (
                _colloc_sampler is not None
                and model_type in EQUATION_CONSTRAINED_MODELS
            ):
                lc = float(hp.get("lambda_collocation", 0.0))
                n_c = int(hp.get("n_collocation", 32))
                if lc > 0.0 and n_c > 0:
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    feat_c, tau_nc = _colloc_sampler.sample(n_c, device)
                    _phy_z = torch.zeros(
                        feat_c.shape[0], ACTIVE_JOINTS, device=device, dtype=feat_c.dtype,
                    )
                    use_amp_c = scaler is not None
                    _raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
                    with torch.autocast(device_type=device.type, enabled=use_amp_c):
                        tau_hat_c, _ = _forward_pass(model, feat_c, _phy_z, model_type)
                        tau_eff = _raw_m.tau_calib(tau_nc)
                        loss_c = lc * F.mse_loss(tau_hat_c, tau_eff)
                    _gc = float(hp.get("grad_clip_norm", 5.0))
                    if scaler is not None:
                        scaler.scale(loss_c).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=_gc)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss_c.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=_gc)
                        optimizer.step()

            val_loss, _val_pred, _val_tgt = eval_epoch(
                model, loaders["val"], device, model_type, hp=hp,
            )
            # Canonical val_rmse: for each trajectory → per-joint RMSE → mean joints → mean trajectories.
            # Normalised space for early-stopping threshold (5e-5 is calibrated here).
            _val_rmse_unw = macro_rmse_numpy(_val_pred, _val_tgt, _val_trajectories)
            # Physical-space val_rmse (N·m) — used for display, history CSV, and physics scheduler.
            _val_pred_phys = _val_pred * _tau_std_val + _tau_mean_val
            _val_tgt_phys  = _val_tgt  * _tau_std_val + _tau_mean_val
            _val_rmse_phys = macro_rmse_numpy(_val_pred_phys, _val_tgt_phys, _val_trajectories)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_rmse"].append(train_rmse_unw)
            history["val_rmse"].append(_val_rmse_phys)   # physical N·m — interpretable in CSV
            history["w_d"].append(_wd_eff)
            history["w_p"].append(_wp_eff)

            if _phys_sched is not None:
                _wd_eff, _wp_eff = _phys_sched.step(epoch, _val_rmse_unw)

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    if _early_metric == "val_loss":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step(_val_rmse_unw)
                elif onecycle_sched is None:
                    scheduler.step()

            _current_lr = optimizer.param_groups[0]["lr"]
            if _early_metric == "val_loss":
                improved = val_loss < (best_val_loss_track - 1e-7)
            else:
                improved = _val_rmse_unw < (best_val_rmse - _min_delta)

            if improved:
                if _early_metric == "val_loss":
                    best_val_loss_track = val_loss
                best_val_loss      = val_loss
                best_val_rmse      = _val_rmse_unw
                best_val_rmse_phys = _val_rmse_phys
                best_epoch         = epoch
                best_state         = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter   = 0
                _status_short      = "[bold green]★ best[/bold green]"
                if _early_metric == "val_rmse" and best_state is not None:
                    try:
                        os.makedirs(_best_ckpt_cache_dir, exist_ok=True)
                        torch.save(
                            {
                                "state_dict": best_state,
                                "epoch": int(epoch),
                                "val_rmse": float(_val_rmse_unw),
                                "model_type": model_type,
                                "model_class": _model_cls_name,
                            },
                            os.path.join(_best_ckpt_cache_dir, "best_val_rmse.pt"),
                        )
                    except OSError as _cache_e:
                        logger.warning("Could not write best val_rmse cache: %s", _cache_e)
            else:
                patience_counter += 1
                remaining    = patience - patience_counter
                _status_short = f"[dim]p={remaining}[/dim]"

            if len(history["val_loss"]) >= 2:
                _delta = val_loss - history["val_loss"][-2]
                if _delta < -1e-6:
                    _trend = f"[green]▼{abs(_delta):.5f}[/green]"
                elif _delta > 1e-6:
                    _trend = f"[red]▲{abs(_delta):.5f}[/red]"
                else:
                    _trend = "[dim]―[/dim]"
            else:
                _trend = "[dim]―[/dim]"

            _pw_str = (
                f"  [dim]w_p={_wp_eff:.3f}[/dim]"
                if model_type in PHYSICS_WEIGHT_MODELS
                else ""
            )
            _best_disp = (
                best_val_rmse_phys if _early_metric == "val_rmse" else best_val_loss_track
            )
            _line = (
                f"  [bright_cyan]{epoch:>4d}[/bright_cyan]/{epochs}  "
                f"[green]train={train_loss:.5f}[/green]  "
                f"[dim]tr_rmse={train_rmse_unw:.4f}[/dim]  "
                f"[yellow]val={val_loss:.5f}[/yellow] {_trend}  "
                f"[dim]va_rmse={_val_rmse_phys:.5f} N·m[/dim]  "
                f"[bright_green]best={_best_disp:.5f} N·m[/bright_green]  "
                f"lr={_current_lr:.2e}{_pw_str}  {_status_short}"
            )
            _epoch_lines.append(_line)

            if device.type == "cuda":
                _gpu_mem_gb = torch.cuda.memory_reserved() / 1e9

            progress.update(
                task, advance=1,
                description=f"[cyan]Epoch {epoch}/{epochs}[/cyan]  "
                            f"[green]train {train_loss:.5f}[/green]  "
                            f"[yellow]val {val_loss:.5f}[/yellow]",
            )
            live.update(_make_live_panel(
                progress, _epoch_lines, epoch, epochs, best_val_rmse_phys,
                gpu_mem_gb=_gpu_mem_gb, gpu_total_gb=gpu_total_gb,
                samples_per_sec=_samples_per_sec, current_lr=_current_lr,
                grad_norm=_grad_norm,
            ))

            if use_early_stop and patience_counter >= patience:
                stopped_early = True
                break

    # Print final epoch summary
    console.print()
    console.print(f"  [bold bright_white]{'Epoch':>8}  {'Train':>11}  {'Val':>11}  "
                  f"{'Best':>11}  Status[/bold bright_white]")
    console.print(f"  [dim]{'─' * 68}[/dim]")
    for _line in _epoch_lines[-10:]:
        console.print(_line)

    if stopped_early:
        console.print(
            f"  [yellow]⏹  Early stopping[/yellow] at epoch [bright_cyan]{epoch}[/bright_cyan]"
            f"  (no improvement on [cyan]{_early_metric}[/cyan] for [yellow]{patience}[/yellow] epochs)"
        )

    elapsed = time.time() - t0
    console.print(
        f"\n  [dim]Training time:[/dim] [bright_cyan]{elapsed:.1f}s[/bright_cyan]  "
        f"  [dim]Best val RMSE:[/dim] [bright_green]{best_val_rmse_phys:.5f} N·m[/bright_green]"
        f"  [dim](normalised {best_val_rmse:.5f})[/dim]  "
        f"  [dim]Best val loss:[/dim] [bright_green]{best_val_loss:.5f}[/bright_green]"
    )

    # Capture final-epoch weights before overwriting with best snapshot.
    final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    final_epoch_trained = len(history["train_loss"])

    # ── Restore best checkpoint ────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation metrics — run on ALL val trajectories (no drop_last), physical units
    _, _final_val_pred, _final_val_tgt = eval_epoch(
        model, loaders["val"], device, model_type, hp=hp,
    )
    _final_val_pred_phys = _final_val_pred * _tau_std_val + _tau_mean_val
    _final_val_tgt_phys  = _final_val_tgt  * _tau_std_val + _tau_mean_val
    val_metrics_final = compute_metrics(_final_val_pred_phys, _final_val_tgt_phys)

    _joint_names_disp = ["J1 yaw", "J2 shoulder", "J3 elbow", "J4 wrist", "J5 wrist roll"]
    _vm_table = Table(
        box=box.ROUNDED, border_style="cyan",
        header_style="bold bright_white on grey23",
        title="[bold cyan]Val Metrics — all val trajectories (physical units)[/bold cyan]",
        padding=(0, 2),
    )
    _vm_table.add_column("Joint",   style="bold white",   min_width=14)
    _vm_table.add_column("RMSE",    style="bright_green", justify="right")
    _vm_table.add_column("MAE",     style="cyan",         justify="right")
    _vm_table.add_column("Max Err", style="dim white",    justify="right")
    _vm_table.add_column("R2",      style="magenta",      justify="right")
    _vm_table.add_column("Pearson", style="blue",         justify="right")
    _vm_table.add_column("NRMSE",   style="yellow",       justify="right")
    for _jj in range(5):
        _nc = "red" if val_metrics_final["nrmse"][_jj] > 0.5 else (
              "yellow" if val_metrics_final["nrmse"][_jj] > 0.25 else "bright_green")
        _rc = "red" if val_metrics_final["r2"][_jj] < 0.5 else (
              "yellow" if val_metrics_final["r2"][_jj] < 0.8 else "bright_green")
        _vm_table.add_row(
            _joint_names_disp[_jj],
            f"{val_metrics_final['rmse'][_jj]:.5f} N·m",
            f"{val_metrics_final['mae'][_jj]:.5f}",
            f"{val_metrics_final['max_error'][_jj]:.4f}",
            f"[{_rc}]{val_metrics_final['r2'][_jj]:.4f}[/{_rc}]",
            f"{val_metrics_final['pearson_r'][_jj]:.4f}",
            f"[{_nc}]{val_metrics_final['nrmse'][_jj]:.4f}[/{_nc}]",
        )
    _vm_table.add_section()
    _vm_table.add_row(
        "[bold]Macro mean[/bold]",
        f"[bold]{val_metrics_final['rmse_mean']:.5f} N·m[/bold]",
        f"[bold]{val_metrics_final['mae_mean']:.5f}[/bold]",
        "—",
        f"[bold]{val_metrics_final['r2_mean']:.4f}[/bold]",
        f"[bold]{val_metrics_final['pearson_r_mean']:.4f}[/bold]",
        f"[bold]{val_metrics_final['nrmse_mean']:.4f}[/bold]",
    )
    _vm_table.add_row(
        "[dim]Pooled (all τ)[/dim]",
        f"[bold cyan]{val_metrics_final['rmse_pooled']:.5f} N·m[/bold cyan]",
        "—", "—",
        f"[bold]{val_metrics_final['r2_overall']:.4f}[/bold]",
        "—", "—",
    )
    console.print()
    console.print(_vm_table)

    # Evaluate on test set
    _, test_pred, test_target = eval_epoch(
        model, loaders["test"], device, model_type, hp=hp,
    )
    _tau_std  = loaders["test"].dataset.std_tau
    _tau_mean = loaders["test"].dataset.mean_tau
    test_pred_phys   = test_pred   * _tau_std + _tau_mean
    test_target_phys = test_target * _tau_std + _tau_mean
    test_metrics = compute_metrics(test_pred_phys, test_target_phys)

    # Metrics table
    joint_names = ["J1 yaw", "J2 shoulder", "J3 elbow", "J4 wrist", "J5 wrist roll"]
    m_table = Table(
        box=box.ROUNDED, border_style="bright_green",
        header_style="bold bright_white on grey23",
        title="[bold bright_green]Test Metrics[/bold bright_green]",
        padding=(0, 2),
    )
    m_table.add_column("Joint",   style="bold white",   min_width=14)
    m_table.add_column("RMSE",    style="bright_green", justify="right")
    m_table.add_column("MAE",     style="cyan",         justify="right")
    m_table.add_column("Max Err", style="dim white",    justify="right")
    m_table.add_column("R2",      style="magenta",      justify="right")
    m_table.add_column("Pearson", style="blue",         justify="right")
    m_table.add_column("NRMSE",   style="yellow",       justify="right")

    for j in range(5):
        nrmse_col = "red" if test_metrics["nrmse"][j] > 0.5 else (
                    "yellow" if test_metrics["nrmse"][j] > 0.25 else "bright_green")
        r2_col    = "red" if test_metrics["r2"][j] < 0.5 else (
                    "yellow" if test_metrics["r2"][j] < 0.8 else "bright_green")
        m_table.add_row(
            joint_names[j],
            f"{test_metrics['rmse'][j]:.5f} N·m",
            f"{test_metrics['mae'][j]:.5f}",
            f"{test_metrics['max_error'][j]:.4f}",
            f"[{r2_col}]{test_metrics['r2'][j]:.4f}[/{r2_col}]",
            f"{test_metrics['pearson_r'][j]:.4f}",
            f"[{nrmse_col}]{test_metrics['nrmse'][j]:.4f}[/{nrmse_col}]",
        )
    m_table.add_section()
    m_table.add_row(
        "[bold]Macro mean[/bold]",
        f"[bold]{test_metrics['rmse_mean']:.5f} N·m[/bold]",
        f"[bold]{test_metrics['mae_mean']:.5f}[/bold]",
        "—",
        f"[bold]{test_metrics['r2_mean']:.4f}[/bold]",
        f"[bold]{test_metrics['pearson_r_mean']:.4f}[/bold]",
        f"[bold]{test_metrics['nrmse_mean']:.4f}[/bold]",
    )
    m_table.add_row(
        "[dim]Pooled (all τ)[/dim]",
        f"[bold bright_green]{test_metrics['rmse_pooled']:.5f} N·m[/bold bright_green]",
        "—",
        "—",
        f"[bold]{test_metrics['r2_overall']:.4f}[/bold]",
        "—",
        "—",
    )
    console.print()
    console.print(m_table)

    # Save model and artefacts
    epochs_trained = len(history["train_loss"])
    mse_val        = test_metrics["mse_pooled"]
    rmse_val       = test_metrics["rmse_pooled"]
    from Neural_Networks.core.checkpoint_io import build_run_id as _build_run_id
    run_id = _build_run_id(model_type, epochs_trained=epochs_trained,
                           rmse=rmse_val, hp=hp)
    # Grid parent processes may set NN_MODELS_DIR_OVERRIDE (env only; no import
    # of grid_search here) so checkpoints land under Trained_Models_GridSearch/.
    # Unset → standard Trained_Models layout.
    _models_dir_eff = os.environ.get("NN_MODELS_DIR_OVERRIDE", "").strip() or MODELS_DIR
    save_dir = os.path.join(_models_dir_eff, MODEL_SAVE_DIRS[model_type], run_id)
    os.makedirs(save_dir, exist_ok=True)

    _train_ds = loaders["train"].dataset
    def _to_list(arr):
        return arr.tolist() if hasattr(arr, 'tolist') else list(arr)
    _norm_stats = {
        "mean_tau": _to_list(_train_ds.mean_tau),
        "std_tau":  _to_list(_train_ds.std_tau),
        "mean_q":   _to_list(_train_ds.mean_q),
        "std_q":    _to_list(_train_ds.std_q),
        # Include velocity and acceleration normalisation so that saved models
        # can be fully reconstructed without access to the original dataset split.
        "mean_qd":  _to_list(_train_ds.mean_qd),
        "std_qd":   _to_list(_train_ds.std_qd),
        "mean_qdd": _to_list(_train_ds.mean_qdd),
        "std_qdd":  _to_list(_train_ds.std_qdd),
    }

    _unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
    _hparams_blob = (
        _unwrapped.hparams if hasattr(_unwrapped, "hparams")
        else getattr(_unwrapped, "config", {})
    )

    from Neural_Networks.core.checkpoint_io import save_checkpoints, exhaustive_hparams
    model_path, final_model_path = save_checkpoints(
        save_dir,
        model=model,
        final_state=final_state,
        best_epoch=int(best_epoch),
        epochs_trained=final_epoch_trained,
        model_cls_name=_model_cls_name,
        hparams_blob=_hparams_blob,
        norm_stats=_norm_stats,
        avg_metrics=test_metrics,
        val_metrics=val_metrics_final,
        test_metrics=test_metrics,
    )

    from Neural_Networks.core.checkpoint_io import dump_yaml
    meta_path = os.path.join(save_dir, "metadata.yaml")
    dump_yaml({
        "model_type":           model_type,
        "run_id":               run_id,
        "data_run_dir":         run_dir,
        "trained_at":           datetime.now().isoformat(),
        "device":               str(device),
        "epochs_trained":       int(epochs_trained),
        "best_epoch":           int(best_epoch),
        "best_val_loss":        float(best_val_loss),
        "best_val_rmse":        float(best_val_rmse),
        "hyperparams":          dict(hp),
        "exhaustive_hyperparams": exhaustive_hparams(
            model_type, hp,
            n_train_samples=int(hp.get("_n_train_samples", 0) or 0),
        ),
        "physics_sched_config": (
            _phys_sched.config_dict() if _phys_sched is not None else None
        ),
        "metrics":              test_metrics,
        "val_metrics":          val_metrics_final,
    }, meta_path)

    save_comparison_plot(test_pred_phys, test_target_phys, test_metrics,
                         os.path.join(save_dir, "comparison_plot.png"), model_type)
    save_architecture_summary(_unwrapped, os.path.join(save_dir, "architecture.txt"))

    # Training history plot + CSV
    hist_path = os.path.join(save_dir, "training_history.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="train loss", color="steelblue")
    ax.plot(history["val_loss"],   label="val loss",   color="darkorange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"Training History — {model_type}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(hist_path, dpi=100)
    plt.close(fig)

    csv_path = os.path.join(save_dir, "training_history.csv")
    _hist_wd = history.get("w_d") or []
    _hist_wp = history.get("w_p") or []
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "train_rmse", "val_rmse", "w_d", "w_p"]
        )
        for i in range(len(history["train_loss"])):
            tl = history["train_loss"][i]
            vl = history["val_loss"][i]
            trm = history["train_rmse"][i]
            vrm = history["val_rmse"][i]
            wd = float(_hist_wd[i]) if i < len(_hist_wd) else 1.0
            wp = float(_hist_wp[i]) if i < len(_hist_wp) else 0.0
            writer.writerow(
                [i + 1, f"{tl:.8f}", f"{vl:.8f}", f"{trm:.8f}", f"{vrm:.8f}", f"{wd:.8f}", f"{wp:.8f}"]
            )

    device_str = (f"cuda:{torch.cuda.get_device_name(0)}"
                  if device.type == "cuda" else "cpu")
    # Skip registry update during grid search (NN_SKIP_REGISTRY=1) — each grid
    # cell is a hyperparameter probe, not a production artefact, so we don't
    # pollute models_registry.yaml with hundreds of rows.
    if os.environ.get("NN_SKIP_REGISTRY", "").strip() not in ("1", "true", "yes"):
        update_registry(
            registry_file      = REGISTRY_FILE,
            model_key          = model_type,
            run_id             = run_id,
            run_dir            = save_dir,
            hp                 = hp,
            metrics            = test_metrics,
            model_path         = model_path,
            training_time_s    = elapsed,
            device_str         = device_str,
            stopped_early      = stopped_early,
            epochs_ran         = epochs_trained,
            epochs_max         = epochs,
            final_train_loss   = history["train_loss"][-1] if history["train_loss"] else 0.0,
            final_val_loss     = history["val_loss"][-1]   if history["val_loss"]   else 0.0,
            num_train_samples  = len(loaders["train"].dataset),
            num_val_samples    = len(loaders["val"].dataset),
        )

    console.print(Panel(
        f"[bold]Saved to:[/bold] [dim]{save_dir}[/dim]\n"
        f"[bold]RMSE (pooled):[/bold]  [bright_green]{test_metrics['rmse_pooled']:.5f} N·m[/bright_green]  "
        f"[dim](macro mean {test_metrics['rmse_mean']:.5f})[/dim]  "
        f"  [bold]NRMSE:[/bold] [yellow]{test_metrics['nrmse_mean']:.4f}[/yellow]  "
        f"  [bold]Time:[/bold] [dim]{_fmt_time(elapsed)}[/dim]",
        title="[bold green]✓  Training Complete[/bold green]",
        border_style="green", padding=(0, 2)
    ))

    # Training metadata — consumed by the Quick-Test-All ranking table in
    # menus.py (epochs column, scheduler column, early-stopped marker, etc.).
    # Keys are prefixed "_" so they don't collide with per-joint metric arrays.
    test_metrics["_epochs_trained"]     = int(epochs_trained)
    test_metrics["_epochs_max"]         = int(epochs)
    test_metrics["_best_epoch"]         = int(best_epoch)
    test_metrics["_stopped_early"]      = bool(stopped_early)
    test_metrics["_best_val_rmse_phys"] = float(best_val_rmse_phys)
    test_metrics["_lr_scheduler"]       = str(hp.get("lr_scheduler", ""))
    test_metrics["_physics_weight"]     = float(hp.get("physics_weight", 0.0))
    test_metrics["_weight_decay"]       = float(hp.get("weight_decay", 0.0))
    return save_dir, test_metrics


# =============================================================================
# BATCH RESULTS SUMMARY
# =============================================================================


def print_batch_results_table(results: list[dict]) -> None:
    """
    Print aggregate test metrics after training multiple models.
    Each entry: model_type, metrics (full dict from compute_metrics), save_dir, time_s.
    """
    if not results:
        return
    console.print()
    console.rule("[bold bright_green] Batch — test metrics summary [/bold bright_green]",
                 style="bright_green")
    sum_table = Table(
        box=box.ROUNDED, border_style="bright_green",
        header_style="bold bright_white on grey23",
        title="[bold bright_green]Test metrics (physical units)[/bold bright_green]\n"
              "[dim]RMSE_p = pooled √(MSE) over all joints×samples (matches val_rmse); "
              "RMSE_m = mean of per-joint RMSEs. R²_ov = one R² on flattened τ.[/dim]",
        padding=(0, 2),
    )
    sum_table.add_column("Model",      style="bold cyan",    min_width=22)
    sum_table.add_column("RMSE_p",     style="bright_green", justify="right", min_width=9)
    sum_table.add_column("RMSE_m",     style="dim",          justify="right", min_width=9)
    sum_table.add_column("R²_ov",      style="magenta",      justify="right", min_width=7)
    sum_table.add_column("R²_m",       style="dim",          justify="right", min_width=7)
    sum_table.add_column("MAE",        style="cyan",         justify="right", min_width=9)
    sum_table.add_column("NRMSE",      style="yellow",       justify="right", min_width=8)
    sum_table.add_column("Pearson",    style="blue",         justify="right", min_width=8)
    sum_table.add_column("Time",       style="dim",          justify="right", min_width=8)

    joint_table = Table(
        box=box.SIMPLE, border_style="dim cyan",
        header_style="bold cyan",
        title="[bold cyan]Per-joint test RMSE (N·m)[/bold cyan]",
        padding=(0, 1),
    )
    joint_table.add_column("Model", style="bold white", min_width=28)
    for j in range(1, 6):
        joint_table.add_column(f"J{j}", justify="right", min_width=9)

    for r in results:
        m  = r["metrics"]
        _rp = m.get("rmse_pooled", m.get("rmse_mean", 0.0))
        _ro = m.get("r2_overall", m.get("r2_mean", 0.0))
        sum_table.add_row(
            r["model_type"],
            f"{_rp:.5f}",
            f"{m['rmse_mean']:.5f}",
            f"{_ro:.4f}",
            f"{m['r2_mean']:.4f}",
            f"{m['mae_mean']:.5f}",
            f"{m['nrmse_mean']:.4f}",
            f"{m['pearson_r_mean']:.4f}",
            _fmt_time(r.get("time_s", 0.0)),
        )
        joint_table.add_row(
            r["model_type"],
            *(f"{m['rmse'][j]:.5f}" for j in range(5)),
        )

    console.print(sum_table)
    console.print()
    console.print(joint_table)

# =============================================================================
# MODEL SELECTION MENU
# =============================================================================

def _build_model_menu():
    """Build a numbered flat list from MODEL_CATEGORIES, returning (list, table)."""
    model_list = []
    for cat_models in MODEL_CATEGORIES.values():
        model_list.extend(cat_models)

    table = Table(
        box=box.ROUNDED, border_style="dim bright_cyan",
        header_style="bold bright_white on grey23",
        padding=(0, 2), show_lines=False,
    )
    table.add_column("#",          style="dim white",  justify="right", min_width=3)
    table.add_column("Category",   style="yellow",     min_width=26)
    table.add_column("Model",      style="bold cyan",  min_width=28)
    table.add_column("Mode",       style="dim white",  min_width=10)
    table.add_column("Description",style="white")

    _desc = {
        "BlackBoxFNN":                ("Pointwise",  "Pure MLP, all 15 kinematic features. No physics."),
        "PhysicsRegularizedFNN":      ("Pointwise",  "FNN + lambda*MSE(tau_hat, tau_phys) in loss."),
        "ResidualCorrectionFNN":      ("Pointwise",  "tau_hat = alpha*tau_phys + delta_tau. Input: [q,qd,qdd,tau_phys]."),
        "LagrangianStructuredFNN":    ("Pointwise",  "4 sub-nets: M(q)qdd + C(q,qd) + g(q) + f(qd). SPD M."),
        "EquationConstrainedPINNFNN": ("Pointwise",  "MLP + physics_loss: MSE(tau_hat - tau_phys, 0)."),
        "DecomposedStructuredPINNFNN":("Pointwise",  "4 component nets + SPD + friction + physics losses."),
    }

    _num = 0
    for cat, cat_models in MODEL_CATEGORIES.items():
        first = True
        for name in cat_models:
            _num += 1
            mode, desc = _desc.get(name, ("", name))
            table.add_row(str(_num), cat if first else "", name, mode, desc)
            first = False

    return model_list, table


def _select_model_types(model_list: list, model_table) -> list[str] | str:
    """
    Prompt for one or more models (comma-separated menu numbers or registry names).
    Returns a non-empty list of model_type strings, or _QUICK_TEST_SENTINEL for option 0.
    """
    console.print()
    console.print(model_table)
    console.print(
        f"  [dim]0.[/dim]  [bold yellow]Quick Test All[/bold yellow]"
        f"  [dim]— train all {len(model_list)} models with default HPs and rank by RMSE[/dim]"
    )
    console.print(
        "  [dim]Enter [bold]one[/bold] menu number or class name, or several separated by commas "
        f"(e.g. [bold]1,4,5[/bold]) to run that subset in sequence — hyperparameters are "
        "prompted once (shared) for the batch, then each run trains in order.[/dim]"
    )
    console.print()
    raw = ask(
        "Select model(s): number(s) or name(s), comma-separated (0 = Quick Test All)",
        default="1",
    ).strip()

    if raw == "0":
        return _QUICK_TEST_SENTINEL

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        console.print("  [red]✗[/red] Empty input after parsing.")
        return _select_model_types(model_list, model_table)

    resolved: list[str] = []
    for p in parts:
        if p.isdigit():
            idx = int(p) - 1
            if not (0 <= idx < len(model_list)):
                console.print(
                    f"  [red]✗[/red] Number [yellow]{p}[/yellow] out of range (valid: 1–{len(model_list)})."
                )
                return _select_model_types(model_list, model_table)
            resolved.append(model_list[idx])
        else:
            if p not in MODEL_REGISTRY:
                console.print(
                    f"  [red]✗[/red] Unknown model [yellow]'{p}[/yellow]. "
                    f"Use a registry name or menu index 1–{len(model_list)}."
                )
                return _select_model_types(model_list, model_table)
            resolved.append(p)

    seen: set[str] = set()
    unique: list[str] = []
    for m in resolved:
        if m not in seen:
            seen.add(m)
            unique.append(m)

    if len(unique) == 1:
        console.print(
            f"  [bold green]✓[/bold green] Selected: [bold bright_cyan]{unique[0]}[/bold bright_cyan]"
        )
    else:
        console.print(
            f"  [bold green]✓[/bold green] Selected [bold]{len(unique)}[/bold] models (in order):"
        )
        for m in unique:
            console.print(f"     [bright_cyan]• {m}[/bright_cyan]")
    return unique


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    # Detect hardware once at startup so every gather_hp() call can
    # inject hardware-appropriate defaults into the prompts.
    hw = detect_hardware()

    console.print()
    console.print(Panel(
        "[bold bright_cyan]Neural Network Torque Prediction[/bold bright_cyan]\n"
        "[dim]Kikobot 6-DOF Robot Arm  ·  12 Models in 5 Categories[/dim]\n\n"
        f"[dim]Hardware:[/dim]  [bold]{hw['profile'].upper()}[/bold]  "
        f"[dim]{hw['gpu_name']}  {hw['vram_gb']} GB VRAM  {hw['ram_gb']:.0f} GB RAM[/dim]\n"
        "[dim]Steps:[/dim]  1 · Model(s) Select (e.g. [cyan]1,4,5[/cyan])    2 · Data Prep    "
        "3 · Hyperparameters (shared + batch)    4 · Train each",
        title="[bold white] PINN TRAINING SUITE [/bold white]",
        border_style="bright_cyan",
        padding=(1, 4),
    ))

    # Step 1: Model Selection (one or many, e.g. 1,4,5)
    section("Model Selection")
    model_list, model_table = _build_model_menu()
    selection = _select_model_types(model_list, model_table)

    if selection == _QUICK_TEST_SENTINEL:
        _run_quick_test_all(model_list, hw=hw)
        console.print()
        console.print(Panel(
            "[bold bright_green]Quick Test All complete![/bold bright_green]\n"
            "[dim]Run [cyan]python -m Neural_Networks.visualizer[/cyan] to compare models.[/dim]",
            border_style="green", padding=(0, 3)
        ))
        console.print()
        return

    assert isinstance(selection, list)
    model_queue: list[str] = selection

    # Step 2: Data Preparation (requirements = union over selected models)
    # Pass TRAIN_DATA_DIR explicitly — cli.menus.data_preparation_step() requires it.
    run_dir, dataset_meta = data_preparation_step(model_queue, TRAIN_DATA_DIR)

    # Step 3: Hyperparameters once for the batch (shared groups), then train each
    hp_by_model = gather_hp_for_models(model_queue, hw=hw, dataset_meta=dataset_meta)

    batch_results: list[dict] = []
    for qi, model_type in enumerate(model_queue, 1):
        console.print()
        console.rule(
            f"[bold bright_cyan] [{qi}/{len(model_queue)}] {model_type} — training [/bold bright_cyan]",
            style="bright_cyan",
        )
        _t0 = time.time()
        _sd, _met = train_model(run_dir, model_type, hp_by_model[model_type])
        batch_results.append({
            "model_type": model_type,
            "metrics":    _met,
            "save_dir":     _sd,
            "time_s":       time.time() - _t0,
        })

    print_batch_results_table(batch_results)

    # Step 5: Offer to train more models on the same dataset
    while True:
        console.print()
        again = ask(
            "Train more model(s) on this dataset? (same or new multi-select)",
            default="no",
            choices=["yes", "no"],
        )
        if again.lower() != "yes":
            break
        section("Model Selection")
        selection = _select_model_types(model_list, model_table)
        if selection == _QUICK_TEST_SENTINEL:
            _run_quick_test_all(model_list, hw=hw)
            break
        assert isinstance(selection, list)
        hp_more = gather_hp_for_models(selection, hw=hw, dataset_meta=dataset_meta)
        more_results: list[dict] = []
        for qi, model_type in enumerate(selection, 1):
            console.print()
            console.rule(
                f"[bold bright_cyan] [{qi}/{len(selection)}] {model_type} — training [/bold bright_cyan]",
                style="bright_cyan",
            )
            _t0 = time.time()
            _sd, _met = train_model(run_dir, model_type, hp_more[model_type])
            more_results.append({
                "model_type": model_type,
                "metrics":    _met,
                "save_dir":     _sd,
                "time_s":       time.time() - _t0,
            })
        print_batch_results_table(more_results)

    console.print()
    console.print(Panel(
        "[bold bright_green]All done![/bold bright_green]\n"
        "[dim]Run [cyan]python -m Neural_Networks.visualizer[/cyan] to compare models.[/dim]",
        border_style="green", padding=(0, 3)
    ))
    console.print()


if __name__ == "__main__":
    main()
