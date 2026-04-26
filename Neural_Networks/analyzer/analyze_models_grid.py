#!/usr/bin/env python3
"""Grid-search analysis -- single interactive entry point.

Usage (from repo root):
    python3 Neural_Networks/analyzer/analyze_models_grid.py

Prompts you to pick a run folder (if multiple exist) and whether to
recompute train metrics, then generates all figures.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Neural_Networks.analyzer.compute.enrich import enrich_records
from Neural_Networks.analyzer.config import _GRID_ROOT
from Neural_Networks.analyzer.io.scan import (
    group_by_model_type,
    list_run_dirs,
    scan_trained_models,
)
from Neural_Networks.analyzer.plots import PLOTS
from Neural_Networks.analyzer.style import setup_plot_style
from Neural_Networks.analyzer.tables.markdown_export import export_summary_markdown
from Neural_Networks.analyzer.tables.summary import print_summary_table

TOP_K = 10

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interactive pickers
# ---------------------------------------------------------------------------

def _pick_run_folder() -> Path:
    """Show available run folders and ask the user to choose one."""
    runs = list_run_dirs(_GRID_ROOT)
    if not runs:
        print(f"No run folders found under {_GRID_ROOT}")
        sys.exit(1)

    if len(runs) == 1:
        print(f"\nAuto-selected the only run folder: {runs[0].name}")
        return runs[0]

    print(f"\nAvailable run folders ({_GRID_ROOT.name}/):\n")
    for i, d in enumerate(runs, 1):
        n = len(list(d.rglob("metadata.yaml")))
        m = re.match(r"^run_(\d{2})(\d{2})_(\d{4})_", d.name)
        date = f"20xx-{m.group(1)}-{m.group(2)}  {m.group(3)[:2]}:{m.group(3)[2:]}" if m else "?"
        print(f"  [{i}]  {d.name}")
        print(f"        {date}   {n} trial(s)")

    print()
    while True:
        try:
            raw = input(f"Select run folder [1-{len(runs)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if raw.isdigit() and 1 <= int(raw) <= len(runs):
            chosen = runs[int(raw) - 1]
            print(f"Selected: {chosen.name}\n")
            return chosen
        # substring match
        matches = [d for d in runs if raw in d.name]
        if len(matches) == 1:
            print(f"Selected: {matches[0].name}\n")
            return matches[0]
        print(f"  Enter a number between 1 and {len(runs)}.")


def _ask_recompute() -> str:
    """Ask whether to recompute train metrics. Returns enrich mode string."""
    print("Train metrics require running model inference on the training split.")
    print("  r = recompute for all models  (slow, ~30 s/model)")
    print("  c = use cached values only    (fast)")
    print("  n = skip train metrics        (fastest, some plots may be incomplete)")
    while True:
        try:
            ans = input("Compute train metrics? [r/c/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "skip"
        if ans in ("r", "recompute"):
            return "all"
        if ans in ("c", "cache", "cached"):
            return "cached"
        if ans in ("", "n", "no", "skip"):
            return "skip"
        print("  Please enter r (recompute), c (use cache), or n (skip).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_dir = _pick_run_folder()
    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = scan_trained_models(str(run_dir))
    if not records:
        logger.error("No trained models found under %s.", run_dir)
        sys.exit(1)

    print(f"Found {len(records)} model(s).\n")
    mode = _ask_recompute()

    print()
    enrich_records(records, mode=mode)
    groups = group_by_model_type(records)
    logger.info("Model types: %s", sorted(groups.keys()))

    print_summary_table(groups)

    setup_plot_style()
    print(f"Saving plots to: {output_dir}\n")

    rendered = 0
    for name, spec in PLOTS.items():
        if not spec.requires(groups):
            logger.info("Skipping %s (requirements not met).", name)
            continue
        kwargs: dict[str, Any] = {}
        if name == "topk_leaderboard":
            kwargs["top_k"] = TOP_K
        try:
            spec.fn(groups, output_dir, **kwargs)
            rendered += 1
        except Exception:
            logger.exception("Plot %s failed.", name)

    export_summary_markdown(groups, output_dir)
    print(f"\nDone. {rendered} figure(s) saved to:\n  {output_dir}")


if __name__ == "__main__":
    main()
