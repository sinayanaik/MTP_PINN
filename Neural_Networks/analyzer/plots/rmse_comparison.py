"""Fig 2 — RMSE comparison (train vs test) and test/val R2 per architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..io.records import (
    arch_short_label, best_per_type,
    rmse_scalar, split_scalar, train_rmse,
)
from ..style import panel_label, type_color_map
from ._common import save_fig


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    all_recs = best_per_type(groups)
    if not all_recs:
        return

    type_colors = type_color_map(list(groups.keys()))
    bar_colors = [type_colors.get(r.get("model_type", "?"), "gray") for r in all_recs]
    labels = [rf"$\mathrm{{{arch_short_label(r.get('model_type', '?'))}}}$" for r in all_recs]

    tr_rmse  = [rmse_scalar(r, "train") for r in all_recs]
    te_rmse  = [rmse_scalar(r, "test") for r in all_recs]
    test_r2  = [split_scalar(r, "test", "r2_overall") for r in all_recs]
    val_r2   = [split_scalar(r, "val",  "r2_overall") for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.32

    fig, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(max(11, n * 3.0), 6.5))

    # ── Panel (a): Train vs Test RMSE ──────────────────────────────────────
    b_train = ax_rmse.bar(x - bw / 2, tr_rmse, bw, color=bar_colors,
                          alpha=0.95, edgecolor="black", linewidth=0.8)
    b_test  = ax_rmse.bar(x + bw / 2, te_rmse, bw, color=bar_colors,
                          alpha=0.50, edgecolor="black", linewidth=0.8, hatch="////")

    for b, v in zip(b_train, tr_rmse):
        if v == v and np.isfinite(v):
            ax_rmse.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0001,
                         rf"${v:.4f}$", ha="center", va="bottom", fontsize=10, fontweight="bold",
                         rotation=45)
    for b, v in zip(b_test, te_rmse):
        if v == v and np.isfinite(v):
            ax_rmse.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0001,
                         rf"${v:.4f}$", ha="center", va="bottom", fontsize=10, fontweight="bold",
                         rotation=45)

    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(labels, rotation=0, ha="center", fontsize=13, fontweight="bold")
    ax_rmse.set_xlabel(r"$\mathrm{Architecture}$", fontsize=14, fontweight="bold")
    ax_rmse.set_ylabel(r"$\mathrm{Average\ RMSE\ (N\cdot m)}$", fontsize=14, fontweight="bold")
    ax_rmse.set_xlim(-0.6, n - 0.4)
    min_rmse = min([v for v in tr_rmse + te_rmse if np.isfinite(v) and v > 0], default=0)
    max_rmse = max([v for v in tr_rmse + te_rmse if np.isfinite(v)], default=1)
    ax_rmse.set_ylim(max(0, min_rmse - 0.02), max_rmse * 1.15)
    ax_rmse.grid(True, axis="y", alpha=0.2)
    panel_label(ax_rmse, "a", fontsize=18, y_offset=-0.16)
    # ── Panel (b): Val R² vs Test R² ──────────────────────────────────────
    b_va_r2 = ax_r2.bar(x - bw / 2, val_r2, bw, color=bar_colors,
                         alpha=0.95, edgecolor="black", linewidth=0.8)
    b_te_r2 = ax_r2.bar(x + bw / 2, test_r2, bw, color=bar_colors,
                         alpha=0.50, edgecolor="black", linewidth=0.8, hatch="////")

    for b, v in zip(b_va_r2, val_r2):
        if v == v and np.isfinite(v):
            ax_r2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0001,
                       rf"${v:.4f}$", ha="center", va="bottom", fontsize=10, fontweight="bold",
                       rotation=45)
    for b, v in zip(b_te_r2, test_r2):
        if v == v and np.isfinite(v):
            ax_r2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0001,
                       rf"${v:.4f}$", ha="center", va="bottom", fontsize=10, fontweight="bold",
                       rotation=45)

    ax_r2.axhline(1.0, color="dimgray", lw=1.0, ls="--", alpha=0.5)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(labels, rotation=0, ha="center", fontsize=13, fontweight="bold")
    ax_r2.set_xlabel(r"$\mathrm{Architecture}$", fontsize=14, fontweight="bold")
    ax_r2.set_ylabel(r"$R^2\ \mathrm{Score}$", fontsize=14, fontweight="bold")
    ax_r2.set_xlim(-0.6, n - 0.4)
    # Static zoomed-in axis to definitively show the PINN improvement
    ax_r2.set_ylim(0.75, 1.05)
    ax_r2.grid(True, axis="y", alpha=0.2)
    panel_label(ax_r2, "b", fontsize=18, y_offset=-0.16)

    # Legend proxies for top legend
    proxy_train = Patch(facecolor="gray", alpha=0.95, edgecolor="black", label=r"$\mathrm{Train\ RMSE\ /\ Val}\ R^2$")
    proxy_test  = Patch(facecolor="gray", alpha=0.50, edgecolor="black", hatch="////", label=r"$\mathrm{Test\ RMSE\ /\ Test}\ R^2$")
    arch_handles = [Patch(facecolor=type_colors[t], label=rf"$\mathrm{{{arch_short_label(t)}}}$", alpha=0.8)
                    for t in sorted(type_colors)]
    
    # Shared top legend
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.legend(handles=arch_handles + [proxy_train, proxy_test],
               loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=len(arch_handles) + 2, fontsize=13,
               framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output_dir / "fig2_rmse_comparison.pdf")
