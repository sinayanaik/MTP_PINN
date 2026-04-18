"""
Neural_Networks.cli.menus
==========================
Interactive menu helpers for the training CLI.

Functions
---------
select_existing_dataset(model_types, train_data_dir)
    → (run_dir, meta) | None  — dataset picker with compact table + detail on select

data_preparation_step(model_types, train_data_dir)
    → (run_dir, meta)         — data prep gate (select existing OR guide to GUI preprocessor)

_build_model_menu()
    → (model_list, Table)     — numbered flat list from MODEL_CATEGORIES

_select_model_types(model_list, model_table)
    → list[str] | QUICK_TEST_SENTINEL — model(s) prompt with validation

print_batch_results_table(results)
    → None                    — aggregate test metrics after multi-model batch

_run_quick_test_all(model_list, train_data_dir, hw)
    → None                    — train all models with default HPs + ranked summary

Constants
---------
QUICK_TEST_SENTINEL      — sentinel string returned by _select_model_types for option-0
"""

from __future__ import annotations

import sys
import time
from typing import Callable

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from Neural_Networks.models import (
    MODEL_CATEGORIES,
    MODEL_REGISTRY,
    DECOMPOSED_MODELS,
    PHYSICS_WEIGHT_MODELS,
)
from Neural_Networks.data.scanner import scan_existing_datasets
from Neural_Networks.tui.console import console, section, subsection
from Neural_Networks.tui.dataset_display import show_dataset_table, print_dataset_summary
from Neural_Networks.cli.prompts import ask
from Neural_Networks.cli.hp_wizard import get_default_hp, gather_hp_for_models
from Neural_Networks.core.trainer import _fmt_time
from Neural_Networks.config.hp_registry import apply_profile_to_hp_dict

# Sentinel returned by _select_model_types when the user picks Quick Test All
QUICK_TEST_SENTINEL: str = "__QUICK_TEST_ALL__"


# =============================================================================
# Dataset selection
# =============================================================================

def select_existing_dataset(
    model_types: str | list[str],
    train_data_dir: str,
) -> tuple[str, dict] | None:
    """Show available preprocessed datasets in a compact table; let the user pick one.

    Parameters
    ----------
    model_types     : one or several model type keys (affects RNEA warning column)
    train_data_dir  : absolute path to the processed datasets folder

    Returns
    -------
    (run_dir, meta_dict)  when a dataset is selected, or ``None`` if user chooses 0.
    """
    datasets = scan_existing_datasets(train_data_dir)
    if not datasets:
        return None

    _mts = [model_types] if isinstance(model_types, str) else list(model_types)
    needs_physics = any(m in (
        DECOMPOSED_MODELS | PHYSICS_WEIGHT_MODELS
    ) for m in _mts)

    show_dataset_table(datasets, _mts, needs_physics=needs_physics)
    console.print("  [dim]Enter [bold]0[/bold] to preprocess new data instead.[/dim]")

    while True:
        raw = Prompt.ask(
            f"[cyan]Select dataset [0-{len(datasets)}][/cyan]",
            console=console, default="1",
        ).strip()
        if raw.isdigit() and 0 <= int(raw) <= len(datasets):
            choice = int(raw)
            break
        console.print(f"    [red]Invalid.[/red] Enter 0–{len(datasets)}")

    if choice == 0:
        return None

    ds      = datasets[choice - 1]
    run_dir = ds["run_dir"]
    print_dataset_summary(ds, run_dir)

    pp       = ds.get("preprocessing", {}) or {}
    has_rnea = bool(pp.get("tau_analytical", {}).get("rnea_enabled", False))
    if any(m in DECOMPOSED_MODELS for m in _mts) and not has_rnea:
        console.print(
            "[bold yellow]Note:[/bold yellow] [dim]This dataset has RNEA disabled. "
            "DecomposedStructuredPINN will train without RNEA gravity prior.[/dim]"
        )

    return run_dir, ds


def data_preparation_step(
    model_types: str | list[str],
    train_data_dir: str,
) -> tuple[str, dict]:
    """Gate step: load an existing dataset or guide the user to the GUI preprocessor.

    If no preprocessed datasets exist the process exits with instructions on how
    to run ``python -m Neural_Networks.preprocess_data``.

    Parameters
    ----------
    model_types     : one or several model type keys
    train_data_dir  : absolute path to the processed datasets folder

    Returns
    -------
    (run_dir, meta_dict)
    """
    section("Data Preparation")

    existing = scan_existing_datasets(train_data_dir)
    if existing:
        _mt_summary = (
            model_types if isinstance(model_types, str) else ", ".join(model_types)
        )
        console.print(Panel(
            f"  [bold]{len(existing)} preprocessed dataset(s)[/bold] found in "
            f"[dim]{train_data_dir}[/dim]\n"
            f"  Planned model(s): [cyan]{_mt_summary}[/cyan]\n"
            "  Select one or create a new dataset via the GUI preprocessor.",
            title="[bold bright_cyan]Dataset Selection[/bold bright_cyan]",
            border_style="bright_cyan", padding=(0, 2),
        ))
        use_existing = ask("Load existing dataset?", default="yes", choices=["yes", "no"])
        if use_existing.lower() in ("yes", "y"):
            result = select_existing_dataset(model_types, train_data_dir)
            if result is not None:
                return result
        console.print()

    # No dataset selected or none available
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
# Model selection menu
# =============================================================================

def _build_model_menu() -> tuple[list[str], Table]:
    """Build a numbered flat list from MODEL_CATEGORIES.

    Iterates the ordered MODEL_CATEGORIES dict so that the menu number matches
    the dict ordering, making the display stable across runs.

    Returns
    -------
    (model_list, rich.Table)
        model_list is a flat list of model_type strings in display order.
    """
    model_list: list[str] = []
    for cat_models in MODEL_CATEGORIES.values():
        model_list.extend(cat_models)

    table = Table(
        box=box.ROUNDED, border_style="dim bright_cyan",
        header_style="bold bright_white on grey23",
        padding=(0, 2), show_lines=False,
    )
    table.add_column("#",           style="dim white",  justify="right", min_width=3)
    table.add_column("Category",    style="yellow",     min_width=26)
    table.add_column("Model",       style="bold cyan",  min_width=28)
    table.add_column("Mode",        style="dim white",  min_width=10)
    table.add_column("Description", style="white")

    _desc: dict[str, tuple[str, str]] = {
        "BlackBoxFNN":                ("Pointwise", "Pure MLP, all 15 kinematic features. No physics."),
        "PhysicsRegularizedFNN":      ("Pointwise", "FNN + lambda*MSE(tau_hat, tau_phys) in loss."),
        "ResidualCorrectionFNN":      ("Pointwise", "tau_hat = alpha*tau_phys + delta_tau. Input: [q,qd,qdd,tau_phys]."),
        "LagrangianStructuredFNN":    ("Pointwise", "4 sub-nets: M(q)qdd + C(q,qd) + g(q) + f(qd). SPD M."),
        "EquationConstrainedPINNFNN": ("Pointwise", "MLP + physics_loss: MSE(tau_hat - tau_phys, 0)."),
        "DecomposedStructuredPINNFNN":("Pointwise", "4 component nets + SPD + friction + physics losses."),
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


def _select_model_types(
    model_list: list[str],
    model_table: Table,
) -> list[str] | str:
    """Prompt for one or more models by menu number or registry name.

    The user may enter a single number, a class name, or a comma-separated mix
    (e.g. ``1,4,5``).  Option ``0`` selects *Quick Test All*.

    Returns
    -------
    list[str]               — ordered list of unique model_type strings, or
    QUICK_TEST_SENTINEL     — when the user enters ``0``
    """
    console.print()
    console.print(model_table)
    console.print(
        f"  [dim]0.[/dim]  [bold yellow]Quick Test All[/bold yellow]"
        f"  [dim]— train all {len(model_list)} models with default HPs and rank by RMSE[/dim]"
    )
    console.print(
        "  [dim]Enter [bold]one[/bold] menu number or class name, or several separated by commas "
        f"(e.g. [bold]1,4,5[/bold]) to run that subset in sequence.[/dim]"
    )
    console.print()

    raw = ask(
        "Select model(s): number(s) or name(s), comma-separated (0 = Quick Test All)",
        default="1",
    ).strip()

    if raw == "0":
        return QUICK_TEST_SENTINEL

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
                    f"  [red]✗[/red] {p} out of range (valid: 1–{len(model_list)})."
                )
                return _select_model_types(model_list, model_table)
            resolved.append(model_list[idx])
        else:
            if p not in MODEL_REGISTRY:
                console.print(
                    f"  [red]✗[/red] Unknown model [yellow]'{escape(p)}'[/yellow]. "
                    f"Use a registry name or menu index 1–{len(model_list)}."
                )
                return _select_model_types(model_list, model_table)
            resolved.append(p)

    # Deduplicate while preserving order
    seen:   set[str]  = set()
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
# Batch results display
# =============================================================================

def print_batch_results_table(results: list[dict]) -> None:
    """Print aggregate inference metrics after training a batch of models.

    The ``metrics`` dict in each result is the AVERAGED (val+test pooled)
    metrics block produced by ``train_model`` — it may carry private
    ``_val_metrics`` and ``_test_metrics`` side-channels so this display can
    show both splits plus their average in a single table.

    Each ``results`` entry must contain:
    ``model_type``, ``metrics``, ``save_dir``, ``time_s``.
    """
    if not results:
        return

    console.print()
    console.rule(
        "[bold bright_green] Batch — inference metrics summary [/bold bright_green]",
        style="bright_green",
    )

    sum_table = Table(
        box=box.ROUNDED, border_style="bright_green",
        header_style="bold bright_white on grey23",
        title=(
            "[bold bright_green]Inference metrics — Val / Test / Avg (physical units)[/bold bright_green]\n"
            "[dim]Inference run on ALL val and test samples.  RMSE_p = pooled "
            "√(MSE); R²_ov = one R² on flattened τ.  Avg row = metrics over "
            "concatenated val+test predictions (not arithmetic mean of scalars).[/dim]"
        ),
        padding=(0, 1),
    )
    sum_table.add_column("Model",   style="bold cyan",    min_width=22)
    sum_table.add_column("Split",   style="dim",          min_width=5)
    sum_table.add_column("N",       style="dim",          justify="right", min_width=7)
    sum_table.add_column("RMSE_p",  style="bright_green", justify="right", min_width=9)
    sum_table.add_column("RMSE_m",  style="dim",          justify="right", min_width=9)
    sum_table.add_column("R²_ov",   style="magenta",      justify="right", min_width=7)
    sum_table.add_column("R²_m",    style="dim",          justify="right", min_width=7)
    sum_table.add_column("MAE",     style="cyan",         justify="right", min_width=9)
    sum_table.add_column("NRMSE",   style="yellow",       justify="right", min_width=8)
    sum_table.add_column("Time",    style="dim",          justify="right", min_width=8)

    joint_table = Table(
        box=box.SIMPLE, border_style="dim cyan",
        header_style="bold cyan",
        title="[bold cyan]Per-joint RMSE (N·m) — averaged val+test[/bold cyan]",
        padding=(0, 1),
    )
    joint_table.add_column("Model", style="bold white", min_width=28)
    for j in range(1, 6):
        joint_table.add_column(f"J{j}", justify="right", min_width=9)

    def _row(name_col: str, split: str, n: int, m: dict,
             time_col: str, emphasis: str = "dim") -> None:
        _rp  = m.get("rmse_pooled", m.get("rmse_mean", 0.0))
        _rm  = m.get("rmse_mean",   0.0)
        _ro  = m.get("r2_overall",  m.get("r2_mean", 0.0))
        _r2m = m.get("r2_mean",     0.0)
        _mae = m.get("mae_mean",    0.0)
        _nr  = m.get("nrmse_mean",  0.0)
        sum_table.add_row(
            name_col, split,
            f"{n:,}" if n else "—",
            f"[{emphasis}]{_rp:.5f}[/{emphasis}]",
            f"{_rm:.5f}",
            f"[{emphasis}]{_ro:.4f}[/{emphasis}]",
            f"{_r2m:.4f}",
            f"{_mae:.5f}",
            f"{_nr:.4f}",
            time_col,
        )

    first = True
    for r in results:
        m   = r["metrics"]                             # avg metrics
        vm  = m.get("_val_metrics")  or {}             # may be empty
        tm  = m.get("_test_metrics") or {}
        n_v = int(m.get("_n_val",  0))
        n_t = int(m.get("_n_test", 0))
        n_a = n_v + n_t
        _tt = _fmt_time(r.get("time_s", 0.0))

        if not first:
            sum_table.add_section()
        first = False

        if vm:
            _row(r["model_type"],           "val", n_v, vm, "—",  "white")
            _row("",                        "test", n_t, tm, "—", "white")
            _row("", "[bold bright_green]avg[/bold bright_green]",
                 n_a, m, _tt, "bold bright_green")
        else:
            _row(r["model_type"], "avg", n_a, m, _tt, "bold bright_green")

        joint_table.add_row(
            r["model_type"],
            *(f"{m['rmse'][j]:.5f}" for j in range(5)),
        )

    console.print(sum_table)
    console.print()
    console.print(joint_table)


# =============================================================================
# Quick Test All
# =============================================================================

def _run_quick_test_all(
    model_list: list[str],
    train_data_dir: str,
    hw: dict | None = None,
    train_model_fn: Callable | None = None,
) -> None:
    """Train every model with default HPs and print a ranked comparison table.

    Parameters
    ----------
    model_list      : ordered list of model type strings from _build_model_menu()
    train_data_dir  : absolute path to the processed datasets folder
    hw              : hardware profile dict from detect_hardware()
    train_model_fn  : callable matching train_model(run_dir, model_type, hp) signature;
                      defaults to Neural_Networks.core.trainer.train_model
    """
    if train_model_fn is None:
        from Neural_Networks.core.trainer import train_model as _train_model
        train_model_fn = _train_model

    section("Quick Test All — Default Hyperparameters")
    console.print(Panel(
        f"[bold]Trains all [bright_cyan]{len(model_list)}[/bright_cyan] models[/bold] "
        "sequentially using [green]default hyperparameters[/green].\n"
        "[dim]• RNEA physics features pre-computed (required by physics models).\n"
        "• All models share the same dataset split.\n"
        "• Results ranked by test RMSE at end.[/dim]",
        border_style="yellow", padding=(0, 2),
        title="[bold yellow]Quick Test All[/bold yellow]",
    ))

    # Data preparation: use RNEA-enabled settings (required by all physics models)
    run_dir, ds_meta = data_preparation_step(
        "PhysicsRegularizedFNN", train_data_dir=train_data_dir,
    )
    n_tr_all = int((ds_meta.get("split") or {}).get("stats", {}).get("train", {}).get("n_samples", 0) or 0)

    # Gather per-model HPs using the same interactive flow as individual model selection
    hp_by_model = gather_hp_for_models(model_list, hw=hw, dataset_meta=ds_meta)

    results: list[dict] = []
    total_t0 = time.time()

    for idx, model_type in enumerate(model_list, 1):
        console.print()
        console.rule(
            f"[bold bright_cyan] [{idx}/{len(model_list)}] {model_type} [/bold bright_cyan]",
            style="bright_cyan",
        )
        hp = hp_by_model.get(model_type, get_default_hp(model_type, n_train_samples=n_tr_all))
        hp["_n_train_samples"] = n_tr_all
        apply_profile_to_hp_dict(model_type, hp)

        t0 = time.time()
        save_dir, metrics = train_model_fn(run_dir, model_type, hp)
        elapsed = time.time() - t0
        results.append({
            "model_type":       model_type,
            "metrics":          metrics,
            "rmse_mean":        metrics["rmse_mean"],
            "rmse_pooled":      metrics.get("rmse_pooled", metrics["rmse_mean"]),
            "r2_overall":       metrics.get("r2_overall", metrics.get("r2_mean", 0.0)),
            "r2_mean":          metrics.get("r2_mean", 0.0),
            "nrmse_mean":       metrics["nrmse_mean"],
            "mse_mean":         metrics["mse_mean"],
            "per_joint":        metrics["rmse"],
            "time_s":           elapsed,
            "save_dir":         save_dir,
            "epochs_trained":   metrics.get("_epochs_trained", 0),
            "epochs_max":       metrics.get("_epochs_max", 0),
            "best_epoch":       metrics.get("_best_epoch", 0),
            "stopped_early":    metrics.get("_stopped_early", False),
            "best_val_rmse":    metrics.get("_best_val_rmse_phys", 0.0),
            "physics_weight":   metrics.get("_physics_weight", 0.0),
        })

    total_elapsed = time.time() - total_t0
    results.sort(key=lambda r: r.get("rmse_pooled", r["rmse_mean"]))

    print_batch_results_table(results)

    # Final ranked summary — averaged (val+test) inference metrics
    console.print()
    console.rule(
        "[bold bright_green] Quick Test All — Final Rankings [/bold bright_green]",
        style="bright_green",
    )
    # Compute relative improvement over worst (BlackBox baseline) for each model
    _worst_rmse = max(r.get("rmse_pooled", r["rmse_mean"]) for r in results)
    rank_table = Table(
        box=box.ROUNDED, border_style="bright_green",
        header_style="bold bright_white on grey23",
        title=(
            "[bold bright_green]All Models — Averaged (val+test) inference metrics, "
            "ranked by pooled RMSE[/bold bright_green]\n"
            "[dim]Inference run on every val and test sample, predictions pooled, "
            "metrics computed once.  † = early stopped.[/dim]"
        ),
        padding=(0, 1),
    )
    rank_table.add_column("#",       style="dim white",     justify="right", min_width=2)
    rank_table.add_column("Model",   style="bold cyan",     min_width=26)
    rank_table.add_column("RMSE_p",  style="bright_green",  justify="right", min_width=9)
    rank_table.add_column("Δ%",      style="bright_yellow", justify="right", min_width=6)
    rank_table.add_column("R²_ov",   style="magenta",       justify="right", min_width=7)
    rank_table.add_column("MAE",     style="dim white",     justify="right", min_width=8)
    rank_table.add_column("NRMSE",   style="yellow",        justify="right", min_width=7)
    rank_table.add_column("Epochs",  style="bright_yellow",  justify="right", min_width=13)
    rank_table.add_column("α (w_p)", style="magenta",       justify="right", min_width=7)
    rank_table.add_column("Time",    style="dim cyan",      justify="right", min_width=7)

    best_p  = results[0].get("rmse_pooled", results[0]["rmse_mean"])
    best_r2 = max(r.get("r2_overall", 0.0) for r in results)
    for rank, r in enumerate(results, 1):
        _rp  = r.get("rmse_pooled", r["rmse_mean"])
        _ro  = r.get("r2_overall",  r.get("r2_mean", 0.0))
        style_rmse = "bold bright_green" if _rp == best_p  else "white"
        style_r2   = "bold bright_green" if _ro == best_r2 else "magenta"
        # Relative improvement vs worst model (usually BlackBox baseline)
        _delta_pct = ((_worst_rmse - _rp) / _worst_rmse * 100) if _worst_rmse > 0 else 0.0
        _delta_str = f"-{_delta_pct:.1f}%" if _delta_pct > 0.05 else "base"
        _delta_style = "bold bright_green" if _delta_pct > 3.0 else ("bright_yellow" if _delta_pct > 0.05 else "dim")
        _ep  = r.get("epochs_trained", 0)
        _epm = r.get("epochs_max", 0)
        _bep = r.get("best_epoch", 0)
        _es  = "†" if r.get("stopped_early", False) else ""
        _pw  = r.get("physics_weight", 0.0) or 0.0
        # α-mixture is only active for PhysicsRegularizedFNN and EquationConstrainedPINNFNN.
        # Structured models (Lagrangian/Decomposed) carry physics in their architecture,
        # so α is irrelevant there — show "—" to avoid confusion.
        _alpha_active = r["model_type"] in {"PhysicsRegularizedFNN", "EquationConstrainedPINNFNN"}
        _mae = r.get("metrics", {}).get("mae_mean", 0.0)
        # Epochs column: best@bep / trained / max  (e.g. "b399/881/1000†")
        if _epm > 0:
            _ep_str = f"b{_bep}/{_ep}/{_epm}{_es}" if _bep > 0 else f"{_ep}/{_epm}{_es}"
        else:
            _ep_str = "—"
        rank_table.add_row(
            str(rank),
            r["model_type"],
            f"[{style_rmse}]{_rp:.5f}[/{style_rmse}]",
            f"[{_delta_style}]{_delta_str}[/{_delta_style}]",
            f"[{style_r2}]{_ro:.4f}[/{style_r2}]",
            f"{_mae:.5f}" if _mae else "—",
            f"{r['nrmse_mean']:.4f}",
            _ep_str,
            f"{_pw:.2f}" if (_pw > 0 and _alpha_active) else "—",
            _fmt_time(r["time_s"]),
        )

    console.print()
    console.print(rank_table)
    _best = results[0]
    console.print(Panel(
        f"[bold bright_green]Best RMSE:[/bold bright_green] "
        f"[bold cyan]{_best['model_type']}[/bold cyan]  "
        f"RMSE_p = [bright_green]{_best.get('rmse_pooled', _best['rmse_mean']):.5f} N·m[/bright_green]  "
        f"R²_ov = [magenta]{_best.get('r2_overall', 0.0):.4f}[/magenta]\n"
        f"[dim]Inference pooled over val+test samples.  "
        f"Total time: {_fmt_time(total_elapsed)}   "
        f"Models trained: {len(results)}[/dim]",
        border_style="bright_green", padding=(0, 2),
    ))
