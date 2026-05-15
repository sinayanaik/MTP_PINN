"""argparse + dispatch for `python -m Neural_Networks.analyzer`.
Also works as `python cli.py` or `python analyzer/cli.py`.
"""
from __future__ import annotations

import argparse
import logging
import sys

try:
    from .config import DEFAULT_MODELS_DIR
    from .io.scan import list_datasets
    from .plots import PLOTS
    from .runner import run_all
except ImportError:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Neural_Networks.analyzer.config import DEFAULT_MODELS_DIR
    from Neural_Networks.analyzer.io.scan import list_datasets
    from Neural_Networks.analyzer.plots import PLOTS
    from Neural_Networks.analyzer.runner import run_all

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse grid-search trained models.\n\n"
            "Models are typically stored as:\n"
            "  Neural_Networks/Trained_Models/Grid_Searches/Trained_Models_Grid_vX/<run>/ModelType/<trial>/\n\n"
            "Pass --models-dir to analyse one run, or omit it to auto-select\n"
            "(single run) or pick from a list."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models-dir", default=DEFAULT_MODELS_DIR,
        help="Run folder to analyse, e.g. Neural_Networks/Trained_Models/Grid_Searches/Trained_Models_Grid/run_0425_1112_...  "
             "Defaults to the grid root (auto-selects if only one run exists).",
    )
    parser.add_argument(
        "--plot", default="all", choices=list(PLOTS) + ["all"],
        help="Render a single plot by name, or 'all' (default).",
    )
    parser.add_argument(
        "--list-plots", action="store_true",
        help="List the available plot names and exit.",
    )
    parser.add_argument(
        "--list-datasets", action="store_true",
        help="List available run folders and exit.",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of rows in the topk_leaderboard plot.",
    )
    parser.add_argument(
        "--palette", default="tab10", choices=("tab10", "colorblind", "Set2"),
        help="Categorical colour palette for architectures (default: tab10).",
    )

    rcg = parser.add_mutually_exclusive_group()
    rcg.add_argument(
        "--recompute", action="store_true",
        help="Force recompute of train metrics for ALL models (overwrites caches).",
    )
    rcg.add_argument(
        "--no-recompute", action="store_true",
        help="Use cached train metrics without prompting (missing → empty).",
    )
    parser.add_argument(
        "--no-train-metrics", action="store_true",
        help="Skip train metrics entirely (no inference, no cache reads).",
    )

    parser.add_argument("--no-plot", action="store_true",
                        help="Print summary table only; render no figures.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures to disk but do not open windows.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    if args.list_plots:
        print("\nAvailable plots:")
        for name in PLOTS:
            print(f"  {name}")
        print()
        return 0

    if args.list_datasets:
        return list_datasets()

    return run_all(args)


if __name__ == "__main__":
    sys.exit(main())
