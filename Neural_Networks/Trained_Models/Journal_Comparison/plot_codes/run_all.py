#!/usr/bin/env python3
"""Generate every Journal_Comparison figure as a PDF.

Usage (from anywhere)::

    python run_all.py                       # all figures
    python run_all.py --only fig03 fig05    # a subset
    python run_all.py --config-override fig_w=8 dpi_save=600

Champion predictions are computed once up front and cached, so the ~10
inference-backed figures share a single forward pass per architecture.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from shared.config import default_config  # noqa: E402


def _discover() -> list[str]:
    return sorted(p.stem for p in HERE.glob("fig*.py"))


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help="figure stems to run (e.g. fig03 fig05)")
    ap.add_argument("--config-override", nargs="*", default=[],
                    metavar="KEY=VAL", help="override PlotConfig fields")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="skip champion cache warm-up")
    ap.add_argument("--no-tables", action="store_true",
                    help="skip CSV table generation")
    args = ap.parse_args()

    overrides = {}
    for kv in args.config_override:
        k, _, v = kv.partition("=")
        overrides[k.strip()] = _coerce(v.strip())

    mods = _discover()
    if args.only:
        wanted = {s.lower() for s in args.only}
        mods = [m for m in mods if any(m.lower().startswith(w) for w in wanted)]
    if not mods:
        print("no matching figures", file=sys.stderr)
        return 2

    if not args.no_prefetch:
        from shared.inference import prefetch_all
        cfg = default_config()
        if overrides:
            cfg = replace(cfg, **overrides)
        print("Prefetching champion predictions (cached) ...")
        prefetch_all(cfg)

    ok, failed = [], []
    for name in mods:
        try:
            mod = importlib.import_module(name)
            cfg = mod.CONFIG
            if overrides:
                cfg = replace(cfg, **overrides)
            path = mod.main(cfg)
            print(f"  [ok]   {name}  ->  {path}")
            ok.append(name)
        except Exception:  # noqa: BLE001
            print(f"  [FAIL] {name}")
            traceback.print_exc()
            failed.append(name)

    print(f"\n{len(ok)}/{len(mods)} figures written to "
          f"{default_config().figures_dir}")

    tables_failed = False
    if not args.no_tables and not args.only:
        try:
            import make_tables
            cfg = default_config()
            if overrides:
                cfg = replace(cfg, **overrides)
            for p in make_tables.main(cfg):
                print(f"  [ok]   table  ->  {p}")
        except Exception:  # noqa: BLE001
            print("  [FAIL] tables")
            traceback.print_exc()
            tables_failed = True

    if failed:
        print("failed: " + ", ".join(failed))
    if failed or tables_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
