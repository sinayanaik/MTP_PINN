"""Fig 17 — test R2 distribution per architecture (box + strip)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..io.records import arch_short_label, split_scalar
from ..style import type_color_map
from ._common import save_fig, zoom_ylim_1d


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    type_colors = type_color_map(list(groups.keys()))
    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN"] if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order and t != "EDR"]

    arch_data: dict[str, list[float]] = {}
    for mtype in arch_order:
        vals = [split_scalar(r, "test", "r2_overall") for r in groups[mtype]]
        vals = [v for v in vals if v == v and np.isfinite(v)]
        if vals:
            arch_data[mtype] = vals

    if not arch_data:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(arch_data) * 3.0), 7))
    x_positions = np.arange(len(arch_data))
    rng = np.random.default_rng(43)

    for xi, (mtype, vals) in enumerate(arch_data.items()):
        c = type_colors.get(mtype, "gray")
        arr = np.array(vals, dtype=np.float64)

        ax.boxplot(
            arr, positions=[xi], widths=0.40, patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=2.5),
            boxprops=dict(facecolor=c, alpha=0.35, linewidth=1.5),
            whiskerprops=dict(linewidth=1.5, color="dimgray"),
            capprops=dict(linewidth=2.0, color="dimgray"),
        )
        jitter = rng.uniform(-0.12, 0.12, size=len(arr))
        ax.scatter(xi + jitter, arr, color=c, s=45, alpha=0.75, zorder=4,
                   edgecolors="white", linewidths=0.6)
        best_val = float(arr.max())
        ax.annotate(
            f"Best: {best_val:.4f}", xy=(xi, best_val), xytext=(xi + 0.25, best_val + 0.0005),
            fontsize=10, color=c, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=c, lw=1.0),
        )

    all_vals = [v for vals in arch_data.values() for v in vals]
    if all_vals:
        zlo, zhi = zoom_ylim_1d(all_vals, min_pad=0.0008, pad_rel=0.1)
        ax.set_ylim(zlo, zhi)
        ax.text(
            0.99, 0.02, f"Zoomed axis  |  full range: [{min(all_vals):.4f}, {max(all_vals):.4f}]",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            color="dimgray", style="italic",
        )

    short_labels = [arch_short_label(t) for t in arch_data]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_labels, fontsize=13, fontweight="bold")
    ax.set_ylabel("Test R² (all grid runs)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Architecture", fontsize=13, fontweight="bold")
    ax.set_title("Test R² Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.35)

    arch_handles = [Patch(facecolor=type_colors.get(t, "steelblue"), label=arch_short_label(t), alpha=0.80)
                    for t in arch_data]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=arch_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(arch_handles), fontsize=11)
    save_fig(fig, output_dir / "fig8_r2_test_distribution.pdf")
