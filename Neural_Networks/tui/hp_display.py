"""
Neural_Networks.tui.hp_display
================================
Rich table for hyperparameter documentation.

The HP docs live in ``Neural_Networks.hp_registry`` (and the copied version
at ``Neural_Networks.config.hp_registry``).  This module owns only the
*display* logic — formatting into a Rich Table and printing.

Usage
-----
    from Neural_Networks.tui.hp_display import print_hp_docs

    docs = get_model_hp_docs("BlackBoxFNN")
    print_hp_docs({**docs[1], **docs[0]})  # common then specific
"""

from __future__ import annotations

from rich import box
from rich.table import Table

from Neural_Networks.tui.console import console


def print_hp_docs(hp_docs: dict) -> None:
    """Print a Rich table listing every hyperparameter's default, options and description.

    Parameters
    ----------
    hp_docs : dict
        Mapping of ``hp_name → {default, choices?, desc, effect}`` as returned by
        ``get_model_hp_docs()`` or ``merge_doc_dicts_for_prompt()``.

    Output columns
    --------------
    Parameter   — key name (bold cyan, no wrap)
    Default     — default value (green)
    Options     — comma-separated choices when applicable (yellow)
    Description + Effect — free-text docs separated by a dim line
    """
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold bright_white on grey23",
        border_style="dim cyan",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Parameter",              style="bold cyan",    no_wrap=True,  min_width=18)
    table.add_column("Default",                style="bright_green", no_wrap=True,  min_width=12)
    table.add_column("Options",                style="yellow",       no_wrap=False, min_width=14)
    table.add_column("Description + Effect",   style="white",        no_wrap=False)

    for name, info in hp_docs.items():
        default = str(info.get("default", "—"))
        choices = info.get("choices", None)
        opts    = ", ".join(choices) if choices else "—"
        body    = f"{info['desc']}\n[dim]{info['effect']}[/dim]"
        table.add_row(name, default, opts, body)

    console.print(table)
