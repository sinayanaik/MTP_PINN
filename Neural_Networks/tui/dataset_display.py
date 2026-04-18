"""
Neural_Networks.tui.dataset_display
=====================================
Rich-styled display helpers for preprocessed dataset information.

All functions in this module write to ``console`` (Rich Console singleton).
They accept plain Python dicts (loaded from metadata.json) and produce
tables / panels for the dataset selection and confirmation steps of the
training workflow.

Pure-logic helpers for the same metadata (sample counts, compact codes,
filter descriptions) live in ``Neural_Networks.data.labels``.

Public API
----------
print_dataset_summary(ds, run_dir)
    Full-detail panel: preprocessing pipeline table + sample counts.

show_dataset_table(datasets, model_types)
    Compact picker table: one row per available dataset.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from rich import box
from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from Neural_Networks.tui.console import console
from Neural_Networks.data.labels import (
    param_tag_from_metadata,
    pipeline_compact_codes,
)
from Neural_Networks.data.labels import (
    preprocessing_pipeline_table_rows,
    train_val_test_split_table_rows,
)


# =============================================================================
# Dataset detail panel
# =============================================================================

def print_dataset_summary(ds: dict, run_dir: str) -> None:
    """Print a full-detail two-table panel for a selected dataset.

    Shows two inner tables:
    1. **Preprocessing pipeline** — one row per quantity/signal, algorithm,
       Savitzky-Golay window, poly-order and edge mode.
    2. **Sample counts by split** — train / val / test row counts with
       configured ratios and the actual fraction of the full dataset.

    This is shown after the user selects a dataset from the picker table so
    they can confirm the filter settings before training begins.
    """
    created = ds.get("created_at", "?")
    try:
        dt_obj  = datetime.fromisoformat(str(created))
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

    # ---- Preprocessing pipeline table --------------------------------------
    try:
        pipe_rows = preprocessing_pipeline_table_rows(ds)
    except Exception:
        pipe_rows = []

    pipe_table = Table(
        box=box.ROUNDED,
        border_style="green",
        header_style="bold white on grey23",
        title="[bold green]Preprocessing pipeline[/bold green] "
              "[dim](shared by train / val / test CSVs)[/dim]",
        padding=(0, 1),
        show_lines=False,
    )
    pipe_table.add_column("Quantity",     style="bold cyan",  min_width=22)
    pipe_table.add_column("Treatment",    style="white",      min_width=28)
    pipe_table.add_column("W",            justify="right",    style="dim", width=5)
    pipe_table.add_column("p",            justify="right",    style="dim", width=4)
    pipe_table.add_column("Mode / notes", style="dim white",  min_width=18)

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
        pipe_table.add_row(
            "—", escape("(preprocessing metadata incomplete)"), "—", "—", "—")

    # ---- Train / val / test split table ------------------------------------
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
    split_table.add_column("Split",        style="bold", justify="left",  width=6)
    split_table.add_column("n samples",    justify="right", style="bright_white", min_width=12)
    split_table.add_column("Config %",     justify="right", style="dim",  min_width=9)
    split_table.add_column("% of all rows",justify="right", style="dim",  min_width=12)

    for r in split_rows:
        split_table.add_row(
            escape(r["split"]),
            escape(r["n_samples"]),
            escape(r["config_ratio"]),
            escape(r["fraction_of_all"]),
        )

    trim_note = (
        f"[dim]End trim: front {escape(trim_f.get('front_pct', '?'))}%  ·  "
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


# =============================================================================
# Dataset picker table
# =============================================================================

def _compact_count(n: Any) -> str:
    """Format a sample count as '12.3k' or '1.2M' for compactness in table cells."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    if n >= 1_000_000:
        x = n / 1_000_000
        s = f"{x:.1f}".rstrip("0").rstrip(".")
        return f"{s}M"
    if n >= 1000:
        x = n / 1000
        s = f"{x:.1f}".rstrip("0").rstrip(".")
        return f"{s}k"
    return str(n)


def show_dataset_table(
    datasets: list[dict],
    model_types: str | list[str],
    *,
    needs_physics: bool = False,
) -> None:
    """Print the compact dataset picker table.

    One row per dataset in ``datasets``.  Columns: pipeline compact codes
    (q / qd / qdd / τm / τ_apf) and per-split sample counts.
    The RNEA column turns red-! for datasets missing physics data when the
    selected model requires RNEA physics features in the input.

    Parameters
    ----------
    datasets     : list of metadata dicts from scan_existing_datasets()
    model_types  : one or more model type names (used to decide RNEA warning)
    needs_physics: pre-computed flag; if True, red '!' for missing-RNEA rows
    """
    t = Table(
        box=box.ROUNDED, border_style="bright_cyan",
        header_style="bold bright_white on grey23", padding=(0, 1),
        title="[bold bright_cyan]Available processed datasets[/bold bright_cyan]\n"
              "[dim]q / qd / qdd / τm = SG window·poly·mode  "
              "(raw | lock→qd | L* = legacy);  τ_apf = analytical post-filter[/dim]",
        show_lines=True,
    )
    t.add_column("#",      style="bold white", justify="right",  width=3)
    t.add_column("Date",   style="cyan",       justify="left",   width=12)
    t.add_column("q",      style="white",      justify="center", width=10)
    t.add_column("qd",     style="white",      justify="center", width=14)
    t.add_column("qdd",    style="white",      justify="center", width=12)
    t.add_column("τm",     style="white",      justify="center", width=10)
    t.add_column("τ_apf",  style="dim",        justify="center", width=10)
    t.add_column("train",  style="dim",        justify="right",  width=8)
    t.add_column("val",    style="dim",        justify="right",  width=7)
    t.add_column("test",   style="dim",        justify="right",  width=7)
    t.add_column("RNEA",   style="green",      justify="center", width=5)

    for idx, ds in enumerate(datasets, start=1):
        try:
            ptag = param_tag_from_metadata(ds)
        except Exception:
            ptag = ds.get("run_name", "?")
        try:
            codes = pipeline_compact_codes(ds)
        except Exception:
            codes = {"q": "?", "qd": "?", "qdd": "?", "tau_m": "?",
                     "rnea": "?", "tau_apf": "?"}

        split_meta = ds.get("split", {}) or {}
        ss         = split_meta.get("stats", {}) or {}

        def _ns(k: str) -> str:
            try:
                n = int(ss.get(k, {}).get("n_trajectories", 0) or 0)
                return str(n) if n else "0"
            except (TypeError, ValueError):
                return "?"

        pp       = ds.get("preprocessing", {}) or {}
        has_rnea = bool(pp.get("tau_analytical", {}).get("rnea_enabled", False))
        if has_rnea:
            rnea_str = "[green]Y[/green]"
        elif needs_physics:
            rnea_str = "[bold red]![/bold red]"
        else:
            rnea_str = "[dim]N[/dim]"

        _rn     = ds.get("run_name", "")
        _rparts = _rn.split("_")
        if len(_rparts) >= 3 and _rparts[1].isdigit() and _rparts[2].isdigit():
            date_str = f"{_rparts[1]} {_rparts[2][:2]}:{_rparts[2][2:]}"
        else:
            date_str = _rn[:10]

        t.add_row(
            str(idx),
            escape(date_str),
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
        "  [dim]# = index · train/val/test = trajectory counts · "
        "After selection, full tables are shown. "
        "RNEA [red]![/red] = needs physics but dataset has no RNEA.[/dim]"
    )
    console.print()
