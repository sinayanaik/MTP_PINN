"""Rich interactive wizard for Neural_Networks.apps.train."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from Neural_Networks.apps.hardware import detect_hardware
from Neural_Networks.apps.hp_registry import (
    GROUP_ORDER,
    HP_KEY_GROUPS,
    apply_accurate_nominal_to_docs,
    apply_profile_to_hp_dict,
    get_model_hp_docs,
    model_needs_group,
    should_prompt_key,
)
from Neural_Networks.apps.scanner import scan_existing_datasets
from Neural_Networks.apps.trainer import train_model
from Neural_Networks.models import MODEL_REGISTRY

_MODEL_CHOICES = tuple(
    m for m in ("PhysicsRegularizedFNN", "ResidualCorrectionFNN") if m in MODEL_REGISTRY
)

_GROUP_TITLE = {
    "global_train": "Global training",
    "fnn_backbone": "FNN backbone",
    "physics": "Physics mixture",
    "residual": "Residual correction",
}


def _train_n_samples(meta: dict) -> int:
    try:
        return int(meta["split"]["stats"]["train"]["n_samples"])
    except (KeyError, TypeError, ValueError):
        return 0


def _inject_hw_into_doc_defaults(all_docs: dict[str, dict[str, Any]], hw: dict) -> None:
    """Seed doc defaults from hardware profile (batch, epochs, hidden widths)."""
    if "batch_size" in all_docs and "batch_size" in hw:
        d = dict(all_docs["batch_size"])
        d["default"] = int(hw["batch_size"])
        all_docs["batch_size"] = d
    if "epochs" in all_docs and "epochs" in hw:
        d = dict(all_docs["epochs"])
        d["default"] = int(hw["epochs"])
        all_docs["epochs"] = d
    if "hidden_layers" in all_docs and "fc_layers" in hw:
        d = dict(all_docs["hidden_layers"])
        d["default"] = list(hw["fc_layers"])
        all_docs["hidden_layers"] = d


def _default_display(value: Any) -> str:
    if isinstance(value, float):
        return repr(value) if abs(value) < 0.01 or value >= 1e6 else str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_user_value(raw: str, default: Any, choices: list[str] | None) -> Any:
    s = raw.strip()
    if s == "":
        return default
    if choices is not None:
        low = {c.lower(): c for c in choices}
        key = s.lower()
        if key in low:
            return low[key]
        if s in choices:
            return s
        raise ValueError(f"Must be one of: {', '.join(choices)}")
    t = type(default)
    if t is bool:
        return s.lower() in ("1", "true", "yes", "y", "on")
    if t is int:
        return int(float(s))
    if t is float:
        return float(s)
    if t is list:
        try:
            v = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            parts = [p.strip() for p in s.split(",") if p.strip()]
            v = [int(float(p)) for p in parts]
        if not isinstance(v, list):
            raise ValueError("Expected a list")
        return v
    return s


def _prompt_one_hp(
    console: Console,
    key: str,
    doc: dict[str, Any],
    shared: dict[str, Any],
) -> Any:
    default = doc["default"]
    choices = doc.get("choices")
    desc = str(doc.get("desc", key))
    effect = str(doc.get("effect", ""))
    body = desc if not effect else f"{desc}\n\n{effect}"
    console.print(Panel(body, title=f"[bold]{key}[/]", border_style="cyan"))
    def_s = _default_display(default)
    ch = f" [{' | '.join(choices)}]" if choices else ""
    while True:
        raw = Prompt.ask(
            f"Value for [bold]{key}[/]{ch} (Enter = default [dim]{def_s}[/])",
            default="",
            console=console,
        )
        try:
            return _parse_user_value(raw, default, choices)
        except (ValueError, TypeError) as e:
            console.print(f"[red]Invalid: {e}. Try again.[/]")


def _gather_hp_interactive(
    console: Console,
    model_type: str,
    n_train_samples: int,
    hw: dict,
    expert: bool,
) -> dict[str, Any]:
    specific, common = get_model_hp_docs(model_type)
    all_docs: dict[str, dict[str, Any]] = {
        k: dict(v) for k, v in {**common, **specific}.items()
    }
    _inject_hw_into_doc_defaults(all_docs, hw)
    _ep = int(all_docs.get("epochs", {}).get("default", 500))
    apply_accurate_nominal_to_docs(
        model_type,
        all_docs,
        n_train_samples=n_train_samples or None,
        epochs=_ep,
    )

    hp: dict[str, Any] = {}
    for group in GROUP_ORDER:
        if not model_needs_group(model_type, group):
            continue
        keys = HP_KEY_GROUPS.get(group, [])
        if not keys:
            continue
        console.print()
        console.print(
            Panel.fit(
                f"[bold]{_GROUP_TITLE.get(group, group)}[/]",
                style="bold green",
            )
        )
        for key in keys:
            if key not in all_docs:
                continue
            if not should_prompt_key(key, expert, hp):
                hp[key] = all_docs[key]["default"]
                continue
            hp[key] = _prompt_one_hp(console, key, all_docs[key], hp)

    ntr = int(n_train_samples or 0)
    hp["_n_train_samples"] = ntr
    apply_profile_to_hp_dict(model_type, hp)
    # apply_profile_to_hp_dict removes this key; trainer metadata still expects it.
    hp["_n_train_samples"] = ntr
    hp.setdefault("stride", int(hw.get("stride", 1)))
    if "torch_compile" not in hp:
        hp["torch_compile"] = bool(hw.get("compile", False))
    return hp


def _pick_dataset(console: Console, train_data_dir: Path) -> dict | None:
    rows = scan_existing_datasets(str(train_data_dir))
    if not rows:
        console.print(
            "[yellow]No processed datasets found under[/] "
            f"[cyan]{train_data_dir}[/].\n"
            "Run preprocessing first (e.g. [bold]python -m Neural_Networks.apps.preprocess[/])."
        )
        return None

    table = Table(title="Processed datasets (newest first)", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Run", style="bold")
    table.add_column("n_train", justify="right")
    table.add_column("Path", overflow="fold")

    for i, meta in enumerate(rows, start=1):
        n_tr = _train_n_samples(meta)
        table.add_row(
            str(i),
            str(meta.get("run_name", "")),
            str(n_tr) if n_tr else "—",
            str(meta.get("run_dir", "")),
        )
    console.print(table)

    choice = IntPrompt.ask(
        "Dataset index",
        default=1,
        console=console,
    )
    if choice < 1 or choice > len(rows):
        console.print("[red]Invalid index.[/]")
        return None
    return rows[choice - 1]


def run_interactive_train(
    *,
    train_data_dir: Path,
    models_dir: Path,
    registry_file: Path,
    nn_dir: Path,
) -> int:
    console = Console()
    console.print(Panel.fit("[bold]Neural_Networks — train[/]", style="bold blue"))

    hw = detect_hardware()
    console.print(
        f"Hardware: [bold]{hw.get('profile')}[/]  "
        f"GPU: {hw.get('gpu_name')}  "
        f"VRAM: {hw.get('vram_gb')} GB\n"
        f"Suggested batch [cyan]{hw.get('batch_size')}[/], "
        f"epochs [cyan]{hw.get('epochs')}[/], "
        f"hidden [cyan]{hw.get('fc_layers')}[/]"
    )

    meta = _pick_dataset(console, train_data_dir)
    if meta is None:
        return 1

    run_dir = str(meta["run_dir"])
    n_train = _train_n_samples(meta)
    console.print(f"\n[green]Selected:[/] [bold]{meta.get('run_name')}[/] → {run_dir}")
    if n_train:
        console.print(f"[dim]Training rows (metadata): {n_train}[/]")

    if len(_MODEL_CHOICES) < 1:
        console.print("[red]No supported model types in MODEL_REGISTRY.[/]")
        return 1

    model_type = Prompt.ask(
        "Model type",
        choices=list(_MODEL_CHOICES),
        default=_MODEL_CHOICES[0],
        show_choices=True,
        console=console,
    )

    expert = Confirm.ask(
        "Expert mode (show advanced hyperparameters, e.g. min_delta)?",
        default=False,
        console=console,
    )

    if not Confirm.ask("Start hyperparameter wizard?", default=True, console=console):
        return 1

    hp = _gather_hp_interactive(
        console, model_type, n_train, hw, expert=expert,
    )

    if not Confirm.ask("Begin training with these settings?", default=True, console=console):
        return 1

    save_dir, metrics = train_model(
        run_dir,
        model_type,
        hp,
        models_dir=str(models_dir),
        registry_file=str(registry_file),
        nn_dir=str(nn_dir),
        console=console,
    )
    console.print(
        f"\n[bold green]Finished.[/] save_dir={save_dir}\n"
        f"rmse_pooled={metrics.get('rmse_pooled')}"
    )
    return 0
