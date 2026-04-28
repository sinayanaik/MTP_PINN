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
    labels = [arch_short_label(r.get("model_type", "?")) for r in all_recs]

    tr_rmse  = [train_rmse(r) for r in all_recs]
    te_rmse  = [rmse_scalar(r, "test") for r in all_recs]
    test_r2  = [split_scalar(r, "test", "r2_overall") for r in all_recs]
    val_r2   = [split_scalar(r, "val",  "r2_overall") for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.28

    fig, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(max(11, n * 3.0), 6.5))

    # ── Panel (a): Train vs Test RMSE ──────────────────────────────────────
    b_train = ax_rmse.bar(x - bw / 2, tr_rmse, bw, color=bar_colors,
                          alpha=0.90, edgecolor="white", linewidth=1.0)
    b_test  = ax_rmse.bar(x + bw / 2, te_rmse, bw, color=bar_colors,
                          alpha=0.60, edgecolor="white", linewidth=1.0, hatch="////")

    for b, v in zip(b_train, tr_rmse):
        if v == v and np.isfinite(v):
            ax_rmse.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                         f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                         rotation=75)
    for b, v in zip(b_test, te_rmse):
        if v == v and np.isfinite(v):
            ax_rmse.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                         f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                         rotation=75)

    valid_rmse = [v for v in tr_rmse + te_rmse if v == v and np.isfinite(v)]
    if valid_rmse:
        ax_rmse.set_ylim(max(0.0, min(valid_rmse) * 0.97), max(valid_rmse) * 1.12)

    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(labels, rotation=0, ha="center", fontsize=12, fontweight="bold")
    ax_rmse.set_xlabel("Architecture", fontsize=13, fontweight="bold")
    ax_rmse.set_ylabel("Average RMSE (N·m)", fontsize=13, fontweight="bold")
    ax_rmse.set_xlim(-0.6, n - 0.4)
    ax_rmse.grid(True, axis="y", alpha=0.35)
    panel_label(ax_rmse, "a")

    # per-panel legend (top, no architecture colours — those go in shared legend)
    proxy_train = Patch(facecolor="gray", alpha=0.90, label="Train RMSE")
    proxy_test  = Patch(facecolor="gray", alpha=0.60, hatch="////", label="Test RMSE")
    ax_rmse.legend(handles=[proxy_train, proxy_test], fontsize=11,
                   loc="upper right", framealpha=0.92, edgecolor="lightgray")

    # ── Panel (b): Test R² vs Val R² ──────────────────────────────────────
    bw2 = 0.28
    b_te_r2 = ax_r2.bar(x - bw2 / 2, test_r2, bw2, color=bar_colors,
                         alpha=0.88, edgecolor="white", linewidth=1.0)
    b_va_r2 = ax_r2.bar(x + bw2 / 2, val_r2, bw2, color=bar_colors,
                         alpha=0.55, edgecolor="white", linewidth=1.0, hatch="xxxx")

    for b, v in zip(b_te_r2, test_r2):
        if v == v and np.isfinite(v):
            ax_r2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                       f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                       rotation=75)
    for b, v in zip(b_va_r2, val_r2):
        if v == v and np.isfinite(v):
            ax_r2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                       f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                       rotation=75)

    all_r2 = [v for v in test_r2 + val_r2 if v == v and np.isfinite(v)]
    if all_r2:
        ax_r2.set_ylim(max(0.0, min(all_r2) - 0.02), min(1.04, max(all_r2) + 0.03))

    ax_r2.axhline(1.0, color="dimgray", lw=1.0, ls="--", alpha=0.5)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(labels, rotation=0, ha="center", fontsize=12, fontweight="bold")
    ax_r2.set_xlabel("Architecture", fontsize=13, fontweight="bold")
    ax_r2.set_ylabel("R²", fontsize=13, fontweight="bold")
    ax_r2.set_xlim(-0.6, n - 0.4)
    ax_r2.grid(True, axis="y", alpha=0.35)
    panel_label(ax_r2, "b")

    proxy_te_r2 = Patch(facecolor="gray", alpha=0.88, label="Test R²")
    proxy_va_r2 = Patch(facecolor="gray", alpha=0.55, hatch="xxxx", label="Val R²")
    ax_r2.legend(handles=[proxy_te_r2, proxy_va_r2], fontsize=11,
                 loc="upper right", framealpha=0.92, edgecolor="lightgray")

    # Shared architecture legend at top-centre
    arch_handles = [Patch(facecolor=type_colors[t], label=arch_short_label(t))
                    for t in sorted(type_colors)]
    fig.tight_layout(rect=[0, 0.0, 1, 0.91])
    fig.legend(handles=arch_handles,
               loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=len(arch_handles), fontsize=11,
               framealpha=0.95, edgecolor="lightgray")

    save_fig(fig, output_dir / "fig2_rmse_comparison.pdf")


