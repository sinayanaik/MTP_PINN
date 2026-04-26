"""Orchestration: scan -> prompt -> enrich -> tables -> plots -> show."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .compute.enrich import enrich_records
from .io.scan import group_by_model_type, resolve_models_dir, scan_trained_models
from .plots import PLOTS
from .prompt import recompute_mode
from .style import setup_plot_style
from .tables.markdown_export import export_summary_markdown
from .tables.summary import print_summary_table

logger = logging.getLogger(__name__)


def _should_show(args: argparse.Namespace) -> bool:
    if getattr(args, "no_show", False) or getattr(args, "no_plot", False):
        return False
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        return False
    return sys.stdout.isatty()


def _select_plot_names(args: argparse.Namespace) -> list[str]:
    requested = getattr(args, "plot", None) or "all"
    if requested == "all":
        return list(PLOTS)
    return [requested]


def run_all(args: argparse.Namespace) -> int:
    """Main entrypoint shared by --plot all and --plot <name>."""
    models_dir = resolve_models_dir(args.models_dir)
    output_dir = Path(models_dir) / "analysis"

    logger.info("Scanning: %s", models_dir)
    try:
        records = scan_trained_models(models_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not records:
        logger.error("No trained models found.  Nothing to report.")
        return 1

    mode = recompute_mode(records, args)
    if mode == "all":
        logger.info("Recomputing train metrics for %d models...", len(records))
    elif mode == "cached":
        logger.info("Using cached train metrics where available (no recompute).")
    else:
        logger.info("Skipping train metrics entirely (no inference, no cache reads).")

    enrich_records(records, mode=mode)
    groups = group_by_model_type(records)
    logger.info("Found %d model(s) in %d type(s): %s",
                len(records), len(groups), sorted(groups.keys()))

    print_summary_table(groups)

    if getattr(args, "no_plot", False):
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving plots to: %s", output_dir)
    setup_plot_style(palette=getattr(args, "palette", "tab10"))

    plot_names = _select_plot_names(args)
    rendered = 0
    for name in plot_names:
        spec = PLOTS[name]
        if not spec.requires(groups):
            logger.info("Skipping %s (requirements not met).", name)
            continue
        try:
            kwargs: dict[str, Any] = {}
            if name == "topk_leaderboard":
                kwargs["top_k"] = getattr(args, "top_k", 10)
            spec.fn(groups, output_dir, **kwargs)
            rendered += 1
        except Exception:
            logger.exception("Plot %s failed.", name)

    export_summary_markdown(groups, output_dir)

    print(f"\nPlots saved to: {output_dir}")
    print(f"{rendered} figure(s) rendered.")

    if _should_show(args):
        plt.show(block=True)

    return 0
