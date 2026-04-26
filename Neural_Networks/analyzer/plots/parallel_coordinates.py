"""Fig 5 — multi-metric parallel coordinates (best-per-type, test metrics only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..io.records import (
    arch_short_label, best_per_type, rmse_scalar, sorted_records, split_scalar,
)
from ..style import type_color_map
from ._common import save_fig


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    all_recs = best_per_type(groups)
    if not all_recs:
        return

    all_recs_full = sorted_records(groups)

    axes_def = [
        ("rmse_mean",  "test",  "Test\nRMSE",  False),
        ("r2_overall", "test",  "Test\nR2",    True),
        ("mae_mean",   "test",  "Test\nMAE",   False),
        ("nrmse_mean", "test",  "Test\nNRMSE", False),
    ]
    n_axes = len(axes_def)
    axis_labels = [a[2] for a in axes_def]

    axis_ranges: dict[str, tuple[float, float]] = {}
    for key, split, lbl, _ in axes_def:
        vals = [split_scalar(r, split, key) for r in all_recs_full]
        valid = [v for v in vals if v == v and np.isfinite(v)]
        axis_ranges[lbl] = (min(valid), max(valid)) if valid else (0.0, 1.0)

    model_rows = []
    for rec in all_recs:
        norm = {}
        raw  = {}
        for key, split, lbl, higher_better in axes_def:
            v = split_scalar(rec, split, key)
            raw[lbl] = v
            mn, mx = axis_ranges[lbl]
            span = mx - mn if mx != mn else 1.0
            if v == v and np.isfinite(v):
                score = (v - mn) / span
                norm[lbl] = score if higher_better else (1.0 - score)
            else:
                norm[lbl] = float("nan")
        model_rows.append({"rec": rec, "raw": raw, "norm": norm,
                           "model_type": rec.get("model_type", "unknown")})

    type_colors = type_color_map(list(groups.keys()))
    x_pos = list(range(n_axes))

    fig, ax = plt.subplots(figsize=(15, 8))

    drawn_types: set[str] = set()
    for d in model_rows:
        mtype = d["model_type"]
        c = type_colors.get(mtype, "steelblue")
        y_vals = [d["norm"].get(lbl, float("nan")) for lbl in axis_labels]
        if any(v != v or not np.isfinite(v) for v in y_vals):
            continue
        ax.plot(x_pos, y_vals, color=c, lw=2.5, alpha=0.85, marker="o", markersize=8)
        ax.text(n_axes - 0.05, y_vals[-1], f"  {arch_short_label(mtype)}",
                fontsize=10, color=c, va="center", fontweight="bold")
        drawn_types.add(mtype)

    for xi in x_pos:
        ax.axvline(xi, color="gray", lw=0.6, alpha=0.4)

    for i, (_, _, lbl, _) in enumerate(axes_def):
        lo, hi = axis_ranges[lbl]
        ax.text(i,  1.07, f"{hi:.4f}", ha="center", va="bottom", fontsize=9, color="#555")
        ax.text(i, -0.09, f"{lo:.4f}", ha="center", va="top",    fontsize=9, color="#555")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(axis_labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalised Score  (1=best across all runs, 0=worst)", fontsize=11)
    ax.set_ylim(-0.15, 1.20)
    ax.grid(axis="y", alpha=0.18)

    legend_handles = [Patch(color=type_colors[t], label=arch_short_label(t))
                      for t in sorted(drawn_types)]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(legend_handles), 6),
            fontsize=10,
        )

    save_fig(fig, output_dir / "fig5_parallel_coordinates.png")
