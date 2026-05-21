#!/usr/bin/env python3
"""Analyse training-history CSVs from diagnostic runs.

Reads training_history.csv from each variant directory under results/,
plots train/val/test curves on a single figure per variant, and prints a
comparison table.

This is the A2 analysis: where best_val vs best_test epochs land tells us
whether early-stop selection on a noisy val curve is the dominant val<test
gap mechanism.

Usage::

    PYTHONPATH=. python3 Neural_Networks/diagnostics/analyze_curves.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

_DIAG_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _DIAG_DIR / "results"


def _find_run_dirs() -> dict[str, Path]:
    """Locate the latest run_dir under each variant subdirectory."""
    runs: dict[str, Path] = {}
    for var_dir in RESULTS_DIR.iterdir():
        if not var_dir.is_dir():
            continue
        if var_dir.name.startswith("_"):
            continue
        sub_runs = [p for p in var_dir.iterdir() if p.is_dir()]
        if not sub_runs:
            continue
        runs[var_dir.name] = max(sub_runs, key=lambda p: p.stat().st_mtime)
    return runs


def _read_history(csv_path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with open(csv_path, newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            for k, v in row.items():
                if k == "epoch":
                    continue
                if v in ("", None):
                    val = float("nan")
                else:
                    try:
                        val = float(v)
                    except ValueError:
                        continue
                cols.setdefault(k, []).append(val)
    return cols


def _argmin_skipna(xs: list[float]) -> int:
    best_i, best_v = -1, float("inf")
    for i, v in enumerate(xs):
        if v == v and v < best_v:
            best_i, best_v = i, v
    return best_i


def main() -> None:
    runs = _find_run_dirs()
    if not runs:
        print("No diagnostic runs found under", RESULTS_DIR)
        return

    summary: list[dict] = []
    fig, axes = plt.subplots(
        len(runs), 1, figsize=(10, 4 * len(runs)), squeeze=False
    )

    for ax, (name, run_dir) in zip(axes[:, 0], sorted(runs.items())):
        hist_csv = run_dir / "training_history.csv"
        if not hist_csv.is_file():
            print(f"[skip] {name}: no training_history.csv at {run_dir}")
            continue
        h = _read_history(hist_csv)

        train_rmse = h.get("train_rmse", [])
        val_rmse   = h.get("val_rmse",   [])
        test_rmse  = h.get("test_rmse",  [])
        ema_val    = h.get("ema_val_rmse", [])

        epochs = list(range(1, len(val_rmse) + 1))
        ax.plot(epochs, train_rmse, label="train", color="steelblue", alpha=0.6)
        ax.plot(epochs, val_rmse,   label="val",   color="darkorange", linewidth=2)
        if test_rmse:
            ax.plot(epochs[: len(test_rmse)], test_rmse, label="test",
                    color="crimson", linewidth=2, linestyle="--")
        if ema_val and any(x == x for x in ema_val):
            ax.plot(epochs[: len(ema_val)], ema_val, label="ema_val",
                    color="seagreen", linewidth=1.5, linestyle=":")

        best_val_i = _argmin_skipna(val_rmse)
        if test_rmse:
            best_test_i = _argmin_skipna(test_rmse)
        else:
            best_test_i = -1
        if best_val_i >= 0:
            ax.axvline(best_val_i + 1, color="darkorange", alpha=0.4,
                       linestyle=":", label=f"best_val ep={best_val_i + 1}")
        if best_test_i >= 0:
            ax.axvline(best_test_i + 1, color="crimson", alpha=0.4,
                       linestyle=":", label=f"best_test ep={best_test_i + 1}")

        ax.set_title(f"{name}  ({run_dir.name})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("trajectory-macro RMSE (N·m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        # Row in summary
        row = {
            "variant": name,
            "epochs": len(val_rmse),
            "best_val_ep": best_val_i + 1 if best_val_i >= 0 else None,
            "best_val_rmse": val_rmse[best_val_i] if best_val_i >= 0 else None,
            "best_test_ep": best_test_i + 1 if best_test_i >= 0 else None,
            "best_test_rmse": test_rmse[best_test_i] if best_test_i >= 0 else None,
            "test@best_val": test_rmse[best_val_i] if test_rmse and best_val_i >= 0 else None,
            "last_val": val_rmse[-1] if val_rmse else None,
            "last_test": test_rmse[-1] if test_rmse else None,
        }
        if ema_val and any(x == x for x in ema_val):
            best_ema_i = _argmin_skipna(ema_val)
            row["best_ema_ep"] = best_ema_i + 1
            row["best_ema_val"] = ema_val[best_ema_i]
            row["test@best_ema"] = test_rmse[best_ema_i] if test_rmse and best_ema_i < len(test_rmse) else None
        summary.append(row)

    fig.suptitle("Phase-A diagnostic — train / val / test RMSE curves")
    fig.tight_layout()
    out_png = RESULTS_DIR / "curves.png"
    fig.savefig(out_png, dpi=110)
    print(f"Saved {out_png}")

    # Tabular summary
    print("\n" + "=" * 100)
    print(f"{'variant':<20} {'epochs':>6} {'best_val_ep':>11} {'best_val':>9} "
          f"{'best_test_ep':>12} {'best_test':>10} {'test@bv':>9} {'gap':>8}")
    print("=" * 100)
    for r in summary:
        gap = (r["test@best_val"] or float("nan")) - (r["best_test_rmse"] or float("nan"))
        print(
            f"{r['variant']:<20} {r['epochs']:>6} {r['best_val_ep']!s:>11} "
            f"{(r['best_val_rmse'] or float('nan')):>9.5f} "
            f"{r['best_test_ep']!s:>12} "
            f"{(r['best_test_rmse'] or float('nan')):>10.5f} "
            f"{(r['test@best_val'] or float('nan')):>9.5f} "
            f"{gap:>+8.5f}"
        )

    out_json = RESULTS_DIR / "phase_a_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
