"""Fig 7 — MAE & NRMSE comparison (train vs test) for best-per-type."""

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
    bar_colors = [type_colors.get(r.get("model_type", "?"), "#888888") for r in all_recs]
    labels = [arch_short_label(r.get("model_type", "?")) for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.32

    fig, (ax_mae, ax_nrmse) = plt.subplots(1, 2, figsize=(max(11, n * 3.2), 6))

    for ax, key, ylabel, title, letter in [
        (ax_mae,   "mae_mean",   "Mean Absolute Error (N·m)", "(a) MAE",   "a"),
        (ax_nrmse, "nrmse_mean", "Normalised RMSE",           "(b) NRMSE", "b"),
    ]:
        tr_v = [split_scalar(r, "train", key) for r in all_recs]
        te_v = [split_scalar(r, "test",  key) for r in all_recs]

        b_tr = ax.bar(x - bw / 2, tr_v, bw, color=bar_colors, alpha=0.90,
                      edgecolor="white", linewidth=0.8)
        b_te = ax.bar(x + bw / 2, te_v, bw, color=bar_colors, alpha=0.60,
                      edgecolor="white", linewidth=0.8, hatch="////")

        for b, v in zip(b_tr, tr_v):
            if v == v and np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.00005,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9, rotation=75)
        for b, v in zip(b_te, te_v):
            if v == v and np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.00005,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9, rotation=75)

        valid = [v for v in tr_v + te_v if v == v and np.isfinite(v)]
        if valid:
            ax.set_ylim(max(0.0, min(valid) * 0.97), max(valid) * 1.10)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Architecture", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(-0.6, n - 0.4)
        panel_label(ax, letter)

    proxy_tr = Patch(facecolor="#888888", alpha=0.90, label="Train")
    proxy_te = Patch(facecolor="#888888", alpha=0.60, hatch="////", label="Test")
    arch_handles = [Patch(facecolor=type_colors[t], label=arch_short_label(t))
                    for t in sorted(type_colors)]
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.legend(handles=arch_handles + [proxy_tr, proxy_te],
               loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(arch_handles) + 2, fontsize=10)

    save_fig(fig, output_dir / "fig7_mae_nrmse_comparison.png")
