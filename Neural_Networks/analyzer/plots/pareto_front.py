"""Fig 13 — Architecture RMSE distribution (box + strip)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..io.records import arch_short_label, rmse_scalar, split_scalar
from ..style import type_color_map
from ._common import save_fig, zoom_ylim_1d


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    type_colors = type_color_map(list(groups.keys()))
    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN"]
                  if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order and t != "EDR"]

    arch_data: dict[str, list[float]] = {}
    for mtype in arch_order:
        vals = [rmse_scalar(r, "test") for r in groups[mtype]]
        vals = [v for v in vals if v == v and np.isfinite(v)]
        if vals:
            arch_data[mtype] = vals

    if not arch_data:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(arch_data) * 3.0), 7))

    x_positions = np.arange(len(arch_data))
    rng = np.random.default_rng(42)

    for xi, (mtype, vals) in enumerate(arch_data.items()):
        c = type_colors.get(mtype, "#888888")
        arr = np.array(vals)

        ax.boxplot(arr, positions=[xi], widths=0.40, patch_artist=True,
                   showfliers=False,
                   medianprops=dict(color="black", linewidth=2.0),
                   boxprops=dict(facecolor=c, alpha=0.35, linewidth=1.2),
                   whiskerprops=dict(linewidth=1.2, color="#444444"),
                   capprops=dict(linewidth=1.5, color="#444444"))

        jitter = rng.uniform(-0.12, 0.12, size=len(arr))
        ax.scatter(xi + jitter, arr, color=c, s=45, alpha=0.75, zorder=4,
                   edgecolors="white", linewidths=0.5)

        best_val = float(arr.min())
        ax.annotate(f"Best: {best_val:.4f}",
                    xy=(xi, best_val), xytext=(xi + 0.25, best_val - 0.0008),
                    fontsize=9, color=c, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.8))

    all_vals = [v for vals in arch_data.values() for v in vals]
    if all_vals:
        zlo, zhi = zoom_ylim_1d(all_vals, min_pad=0.0004, pad_rel=0.12)
        ax.set_ylim(zlo, zhi)
        ax.text(
            0.99, 0.98,
            f"Y-axis: zoomed to run spread  |  global min/max: [{min(all_vals):.4f}, {max(all_vals):.4f}] N·m",
            transform=ax.transAxes, fontsize=8, ha="right", va="top", color="#444444", style="italic",
        )

    short_labels = [arch_short_label(t) for t in arch_data]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Average RMSE (N·m)", fontsize=12)
    ax.set_xlabel("Architecture", fontsize=12)
    ax.grid(True, axis="y", alpha=0.35)

    arch_handles = [Patch(facecolor=type_colors.get(t, "#888888"),
                          label=arch_short_label(t), alpha=0.80)
                    for t in arch_data]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=arch_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=len(arch_handles), fontsize=10)

    save_fig(fig, output_dir / "fig13_rmse_distribution.png")
