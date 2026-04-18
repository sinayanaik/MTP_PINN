#!/usr/bin/env python3
"""Headless dataset builder (replaces GUI preprocessor)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_APPS = Path(__file__).resolve().parent
_NN_DIR = _APPS.parent.parent
_PROJECT_ROOT = _NN_DIR.parent


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("preprocess")

    p = argparse.ArgumentParser(
        description="Build a preprocessed dataset under Neural_Networks/train_data/",
    )
    p.add_argument(
        "--raw-dir",
        default=str(_PROJECT_ROOT / "raw_samples"),
        help="Directory containing trajectory .json files",
    )
    p.add_argument(
        "--run-dir",
        required=True,
        help="Output run directory (e.g. .../train_data/run_<tag>/)",
    )
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--trim-front-pct", type=float, default=0.0)
    p.add_argument("--trim-back-pct", type=float, default=0.0)
    p.add_argument("--split-mode", choices=("stratified", "random", "temporal"), default="stratified")
    p.add_argument("--q-smooth", action="store_true", help="Savitzky-Golay smooth on q")
    p.add_argument("--q-window", type=int, default=15)
    p.add_argument("--q-polyorder", type=int, default=3)
    p.add_argument("--deriv-window", type=int, default=25)
    p.add_argument("--deriv-polyorder", type=int, default=3)
    p.add_argument("--deriv-mode", default="interp")
    p.add_argument("--qdd-locked", action="store_true", default=True)
    p.add_argument("--qdd-unlocked", action="store_true", help="Use separate SG for qdd")
    p.add_argument("--qdd-window", type=int, default=25)
    p.add_argument("--qdd-polyorder", type=int, default=3)
    p.add_argument("--qdd-mode", default="interp")
    p.add_argument("--tau-smooth", action="store_true")
    p.add_argument("--tau-window", type=int, default=15)
    p.add_argument("--tau-polyorder", type=int, default=3)
    p.add_argument("--no-rnea", action="store_true", help="Disable RNEA (default: RNEA on)")
    p.add_argument("--tau-postfilter", action="store_true")
    p.add_argument("--tau-pf-window", type=int, default=15)
    p.add_argument("--tau-pf-polyorder", type=int, default=3)
    args = p.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        log.error("train + val + test ratios must sum to 1.0")
        return 1

    qdd_locked = not args.qdd_unlocked
    use_rnea = not args.no_rnea

    params: dict = {
        "raw_dir": args.raw_dir,
        "run_dir": args.run_dir,
        "split_mode": args.split_mode,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "trim_front_pct": args.trim_front_pct,
        "trim_back_pct": args.trim_back_pct,
        "q_smooth_enabled": args.q_smooth,
        "q_window": args.q_window,
        "q_polyorder": args.q_polyorder,
        "deriv_window": args.deriv_window,
        "deriv_polyorder": args.deriv_polyorder,
        "deriv_mode": args.deriv_mode,
        "qdd_locked": qdd_locked,
        "tau_smooth_enabled": args.tau_smooth,
        "tau_window": args.tau_window,
        "tau_polyorder": args.tau_polyorder,
        "use_rnea": use_rnea,
        "tau_ana_postfilter_enabled": args.tau_postfilter,
        "tau_ana_window": args.tau_pf_window,
        "tau_ana_polyorder": args.tau_pf_polyorder,
        "geom_config": {},
    }
    if not qdd_locked:
        params["qdd_window"] = args.qdd_window
        params["qdd_polyorder"] = args.qdd_polyorder
        params["qdd_mode"] = args.qdd_mode

    from Neural_Networks.apps.preprocess_core import _do_build

    try:
        result = _do_build(params)
    except Exception as exc:
        log.exception("Build failed: %s", exc)
        return 1

    for line in result.get("log_lines", []):
        log.info("%s", line)
    log.info("Done -> %s", result.get("meta_path", "?"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
