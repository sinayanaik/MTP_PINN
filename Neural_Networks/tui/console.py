"""
Neural_Networks.tui.console
============================
Rich Console singleton and section-heading helpers.

Consuming modules should import ``console`` from here to ensure all Rich
output goes to the same Console instance (same stderr/stdout target).

Usage
-----
    from Neural_Networks.tui.console import console, section, subsection

    section("Model Selection")
    subsection("Choose architecture")
    console.print("[cyan]Hello![/cyan]")
"""

from __future__ import annotations

from rich.console import Console
from rich.rule import Rule

# ---------------------------------------------------------------------------
# Shared Console singleton
# All modules that emit Rich output should use this instance so that styles,
# width settings, and redirect patchwork stay consistent throughout a run.
# ---------------------------------------------------------------------------
console: Console = Console()


def section(title: str) -> None:
    """Print a prominent full-width horizontal rule with a title in CAPITALS.

    Visually separates the major phases of the training workflow
    (Model Selection, Data Preparation, Hyperparameter Configuration, etc.).
    """
    console.print()
    console.rule(
        f"[bold bright_cyan] {title.upper()} [/bold bright_cyan]",
        style="bright_cyan",
    )


def subsection(title: str) -> None:
    """Print a smaller inline heading for a sub-step within a section.

    E.g. sub-steps within the HP configuration section ("Shared — EarlyStopping").
    """
    console.print(f"\n  [bold yellow]── {title} ──[/bold yellow]")
