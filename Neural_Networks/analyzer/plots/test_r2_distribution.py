"""Fig 17 — test R2 distribution per architecture (box + strip)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..io.records import arch_short_label, split_scalar
from ..style import panel_label, type_color_map
from ._common import save_fig


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    type_colors = type_color_map(list(groups.keys()))
    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN"] if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order and t != "EDR"]

    arch_data: dict[str, list[float]] = {}
    for mtype in arch_order:
        vals = [split_scalar(r, "test", "r2_overall") for r in groups[mtype]]
        vals = [v for v in vals if v == v and np.isfinite(v)]
        if vals:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            filtered = [v for v in vals if lower <= v <= upper]
            if filtered:
                arch_data[mtype] = filtered

    if not arch_data:
        return

    fig, ax = plt.subplots(figsize=(max(9, len(arch_data) * 2.5), 7))
    x_positions = np.arange(len(arch_data))
    rng = np.random.default_rng(43)

    for xi, (mtype, vals) in enumerate(arch_data.items()):
        c = type_colors.get(mtype, "gray")
        arr = np.array(vals, dtype=np.float64)

        ax.boxplot(
            arr, positions=[xi], widths=0.45, patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=2.0),
            boxprops=dict(facecolor=c, alpha=0.4, linewidth=1.2),
            whiskerprops=dict(linewidth=1.2, color="dimgray"),
            capprops=dict(linewidth=1.2, color="dimgray"),
        )
        jitter = rng.uniform(-0.15, 0.15, size=len(arr))
        ax.scatter(xi + jitter, arr, color=c, s=40, alpha=0.6, zorder=4,
                   edgecolors="none")
        
        best_val = float(arr.max())
        ax.annotate(
            rf"$\mathrm{{Best:}}\ {best_val:.4f}$", xy=(xi, best_val), 
            xytext=(0, 10), textcoords="offset points",
            fontsize=10, color=c, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2, mutation_scale=12),
        )

    short_labels = [rf"$\mathrm{{{arch_short_label(t)}}}$" for t in arch_data]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$R^2\ \mathrm{Score}$", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    
    all_vals = [v for vals in arch_data.values() for v in vals]
    if all_vals:
        ymin, ymax = min(all_vals), max(all_vals)
        margin = max((ymax - ymin) * 0.05, 0.005)
        ax.set_ylim(ymin - margin, ymax + margin * 1.5)

    arch_handles = [Patch(facecolor=type_colors.get(t, "steelblue"), label=rf"$\mathrm{{{arch_short_label(t)}}}$", alpha=0.8)
                    for t in arch_data]
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(handles=arch_handles, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=len(arch_handles), fontsize=11, frameon=True)
               
    save_fig(fig, output_dir / "fig8_r2_test_distribution.pdf")
