"""Fig 16 — train vs test generalization gap (best-per-type)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..io.records import arch_short_label, best_per_type, rmse_scalar, split_scalar
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
    tr_rmse = [rmse_scalar(r, "train") for r in all_recs]
    te_rmse = [rmse_scalar(r, "test")  for r in all_recs]
    tr_r2   = [split_scalar(r, "train", "r2_overall")   for r in all_recs]
    te_r2   = [split_scalar(r, "test",  "r2_overall")   for r in all_recs]

    gap_rmse = [(t - s) if (s == s and t == t) else float("nan") for s, t in zip(tr_rmse, te_rmse)]
    gap_r2   = [(a - b) if (a == a and b == b) else float("nan") for a, b in zip(tr_r2, te_r2)]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(max(10, n * 2.4), 5.0))

    b0 = ax0.bar(x, gap_rmse, 0.55, color=bar_colors, alpha=0.88, edgecolor="white", linewidth=0.6)
    ax0.axhline(0.0, color="#333333", lw=1.0)
    for b, g in zip(b0, gap_rmse):
        if g == g and np.isfinite(g):
            h = b.get_height()
            ax0.text(
                b.get_x() + b.get_width() / 2, h + 0.0015 * (np.sign(h) or 1),
                f"{g:+.4f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=8, fontweight="bold",
            )
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax0.set_ylabel("Test RMSE - Train RMSE (N·m)", fontsize=10)
    ax0.set_xlabel("Architecture (best-per-type run)", fontsize=11)
    ax0.set_xlim(-0.6, n - 0.4)
    ax0.set_title("Generalization gap -- Average RMSE", fontsize=12, fontweight="bold")
    ax0.grid(True, axis="y", alpha=0.35)
    panel_label(ax0, "a")
    gfin = [g for g in gap_rmse if g == g and np.isfinite(g)]
    if gfin:
        ax0.text(0.02, 0.95, f"min={min(gfin):.4f}  max={max(gfin):.4f} N·m", transform=ax0.transAxes,
                 fontsize=8, va="top", color="#555555", style="italic")

    b1 = ax1.bar(x, gap_r2, 0.55, color=bar_colors, alpha=0.88, edgecolor="white", linewidth=0.6)
    ax1.axhline(0.0, color="#333333", lw=1.0)
    for b, g in zip(b1, gap_r2):
        if g == g and np.isfinite(g):
            h = b.get_height()
            ax1.text(
                b.get_x() + b.get_width() / 2, h + 0.001 * (np.sign(h) or 1),
                f"{g:+.4f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=8, fontweight="bold",
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Train R² - Test R²", fontsize=10)
    ax1.set_xlabel("Architecture (best-per-type run)", fontsize=11)
    ax1.set_xlim(-0.6, n - 0.4)
    ax1.set_title("Generalization gap -- R² overall", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.35)
    panel_label(ax1, "b")

    fig.tight_layout()
    save_fig(fig, output_dir / "fig16_train_test_generalization_gap.png")
