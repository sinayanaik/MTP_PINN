"""Recompute-prompt UX. Called once at startup, before enrichment."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Literal

from .compute.train_metrics import has_cache

logger = logging.getLogger(__name__)

Mode = Literal["all", "cached", "skip"]


def recompute_mode(records: list[dict[str, Any]], args: argparse.Namespace) -> Mode:
    """Decide whether to recompute training metrics.

    Resolution order:
        1. CLI overrides: --no-train-metrics / --recompute / --no-recompute
        2. Non-TTY: silent — recompute everything iff any cache is missing,
           else use caches.
        3. TTY: status line + simple y/N prompt.
    """
    if getattr(args, "no_train_metrics", False):
        return "skip"
    if getattr(args, "recompute", False):
        return "all"
    if getattr(args, "no_recompute", False):
        return "cached"

    n = len(records)
    if n == 0:
        return "cached"

    cached = sum(1 for r in records if has_cache(r))
    missing = n - cached

    if not sys.stdin.isatty():
        return "all" if missing > 0 else "cached"

    print(f"\nCached: {cached}/{n}.  Missing: {missing}/{n}.")
    try:
        ans = input(f"Recompute training metrics for all {n} models? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "cached"
    return "all" if ans.startswith("y") else "cached"
