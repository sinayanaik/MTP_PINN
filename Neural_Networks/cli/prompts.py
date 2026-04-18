"""
Neural_Networks.cli.prompts
=============================
Thin Rich-styled wrappers around ``rich.prompt.Prompt``.

``ask()`` and ``ask_list()`` are used by every CLI wizard step: model
selection, HP configuration, and dataset selection.  They apply consistent
styling (cyan prompt, red error feedback, dim default/choice hints) with
type coercion and choice validation built in.

Usage
-----
    from Neural_Networks.cli.prompts import ask, ask_list

    learning_rate = ask("learning_rate", default=1e-4, cast=float)
    layers        = ask_list("hidden_layers", default=[256, 128], cast=int)
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from rich.prompt import Prompt

from Neural_Networks.tui.console import console


def ask(
    prompt: str,
    default=None,
    cast=str,
    choices=None,
) -> Any:
    """Rich-styled interactive prompt with type coercion and choice validation.

    Press Enter to accept the default.  The default is shown in brackets after
    the prompt; it is highlighted if it matches one of the choices.

    Parameters
    ----------
    prompt  : str   Prompt label (shown after a cyan ▸ prefix).
    default :       If provided, shown as default; Enter accepts it.
    cast    :       Callable to convert the raw string input (e.g. int, float).
    choices : list  When provided, both the literal strings and 1-based numeric
                    indices are valid inputs.

    Returns
    -------
    The casted value (same type as ``cast(raw_input)``).
    """
    choices_hint = (
        "  [dim]("
        + "  ".join(
            f"[bold white]{c}[/bold white]"
            if str(c) == str(default) else str(c)
            for c in choices
        )
        + ")[/dim]"
    ) if choices else ""

    display      = f"[cyan]▸ {prompt}[/cyan]{choices_hint}"
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
            console.print(
                f"    [red]✗[/red] Expected [yellow]{cast.__name__}[/yellow], "
                f"got [red]'{escape(raw)}'[/red]"
            )
            continue

        if choices:
            # Support 1-based numeric index for list choices
            if str(raw).isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            raw_norm   = str(raw).lower()
            val_norm   = str(val).lower()
            valid_norms = [str(c).lower() for c in choices]
            if raw_norm not in valid_norms and val_norm not in valid_norms:
                opts = "  ".join(
                    f"[bold white]{c}[/bold white]"
                    if str(c).lower() == raw_norm else str(c)
                    for c in choices
                )
                console.print(f"    [red]✗[/red] Choose: {opts}")
                continue

        return val


def ask_list(prompt: str, default: list, cast=int) -> list:
    """Prompt for a comma-separated list of values; Enter accepts the default.

    Example — prompting for hidden layer sizes::

        layers = ask_list("hidden_layers", default=[256, 128], cast=int)

    The raw string ``"256,128"`` is converted to ``[256, 128]``.
    """
    default_str = ",".join(str(v) for v in default)
    while True:
        raw = Prompt.ask(
            f"[cyan]▸ {prompt}[/cyan]",
            console=console, default=default_str,
        ).strip()

        if not raw or raw == default_str:
            return default

        try:
            return [cast(v.strip()) for v in raw.split(",")]
        except ValueError:
            console.print(
                f"    [red]✗[/red] Comma-separated [yellow]{cast.__name__}[/yellow] "
                f"values expected, e.g. [dim]{default_str}[/dim]"
            )
