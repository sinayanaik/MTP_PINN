"""Fig 4 — MAE & NRMSE comparison (train vs test) for best-per-type."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..io.records import arch_short_label, best_per_type, split_scalar
from ..style import panel_label, type_color_map
from ._common import save_fig


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    all_recs = best_per_type(groups)
    if not all_recs:
        return

    type_colors = type_color_map(list(groups.keys()))
    bar_colors = [type_colors.get(r.get("model_type", "?"), "gray") for r in all_recs]
    labels = [rf"$\mathrm{{{arch_short_label(r.get('model_type', '?'))}}}$" for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.32

    fig, (ax_mae, ax_nrmse) = plt.subplots(1, 2, figsize=(max(11, n * 3.2), 6.5))

    for ax, key, ylabel, title, letter in [
        (ax_mae,   "mae_mean",   r"$\mathrm{Mean\ Absolute\ Error\ (N\cdot m)}$", r"$\mathrm{MAE\ per\ Architecture}$",   "a"),
        (ax_nrmse, "nrmse_mean", r"$\mathrm{Normalised\ RMSE}$",           r"$\mathrm{NRMSE\ per\ Architecture}$", "b"),
    ]:
        tr_v = [split_scalar(r, "train", key) for r in all_recs]
        te_v = [split_scalar(r, "test",  key) for r in all_recs]

        b_tr = ax.bar(x - bw / 2, tr_v, bw, color=bar_colors, alpha=1.0,
                      edgecolor="black", linewidth=1.0, label="Train")
        b_te = ax.bar(x + bw / 2, te_v, bw, color=bar_colors, alpha=0.45,
                      edgecolor="black", linewidth=1.0, hatch="////", label="Test")

        for b, v in zip(b_tr, tr_v):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0001,
                        rf"${v:.4f}$", ha="center", va="bottom", fontsize=10,
                        fontweight="bold", rotation=45)
            elif v == 0:
                 ax.text(b.get_x() + b.get_width() / 2, 0.001,
                        r"$\mathrm{N/A}$", ha="center", va="bottom", fontsize=9, color="gray")

        for b, v in zip(b_te, te_v):
            if v == v and np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0001,
                        rf"${v:.4f}$", ha="center", va="bottom", fontsize=10,
                        fontweight="bold", rotation=45)

        valid = [v for v in tr_v + te_v if v == v and np.isfinite(v) and v > 0]
        if valid:
            min_v = min(valid)
            ax.set_ylim(max(0, min_v - (min_v * 0.2)), max(valid) * 1.25)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=13, fontweight="bold")
        # ax.set_xlabel removed
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.set_xlim(-0.6, n - 0.4)
        ax.grid(True, axis="y", alpha=0.25)
        panel_label(ax, letter, fontsize=18, y_offset=-0.16)

    proxy_tr = Patch(facecolor="gray", alpha=1.0, edgecolor="black", label=r"$\mathrm{Train}$")
    proxy_te = Patch(facecolor="gray", alpha=0.45, edgecolor="black", hatch="////", label=r"$\mathrm{Test}$")
    arch_handles = [Patch(facecolor=type_colors[t], label=rf"$\mathrm{{{arch_short_label(t)}}}$", alpha=0.8)
                    for t in sorted(type_colors)]
    
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.legend(handles=arch_handles + [proxy_tr, proxy_te],
               loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=len(arch_handles) + 2, fontsize=15,
               framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output_dir / "fig4_mae_nrmse_comparison.pdf")
