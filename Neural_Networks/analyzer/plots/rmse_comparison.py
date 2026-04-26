"""Fig 2 — RMSE comparison (train vs test) + zoomed test + delta vs Black-Box."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..io.records import (
    arch_short_label, best_blackbox_record, best_per_type,
    rmse_scalar, split_scalar, train_rmse,
)
from ..style import panel_label, type_color_map
from ._common import save_fig, zoom_ylim_1d


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    all_recs = best_per_type(groups)
    if not all_recs:
        return

    type_colors = type_color_map(list(groups.keys()))
    bar_colors = [type_colors.get(r.get("model_type", "?"), "#888888") for r in all_recs]
    labels = [arch_short_label(r.get("model_type", "?")) for r in all_recs]

    tr_rmse = [train_rmse(r) for r in all_recs]
    te_rmse = [rmse_scalar(r, "test") for r in all_recs]
    test_r2 = [split_scalar(r, "test", "r2_overall") for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.32

    fig, axes = plt.subplots(2, 2, figsize=(max(11, n * 2.9), 10.0))
    ax_rmse, ax_r2, ax_z, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel (a): Train vs Test RMSE
    b_train = ax_rmse.bar(x - bw / 2, tr_rmse, bw, color=bar_colors,
                          alpha=0.90, edgecolor="white", linewidth=0.8)
    b_test  = ax_rmse.bar(x + bw / 2, te_rmse, bw, color=bar_colors,
                          alpha=0.60, edgecolor="white", linewidth=0.8, hatch="////")

    for b, v in zip(b_train, tr_rmse):
        if v == v and np.isfinite(v):
            ax_rmse.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                         f"{v:.4f}", ha="center", va="bottom", fontsize=9, rotation=75)
    for b, v in zip(b_test, te_rmse):
        if v == v and np.isfinite(v):
            ax_rmse.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                         f"{v:.4f}", ha="center", va="bottom", fontsize=9, rotation=75)

    valid_rmse = [v for v in tr_rmse + te_rmse if v == v and np.isfinite(v)]
    if valid_rmse:
        ax_rmse.set_ylim(max(0.0, min(valid_rmse) * 0.97), max(valid_rmse) * 1.08)

    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(labels, rotation=0, ha="center", fontsize=11, fontweight="bold")
    ax_rmse.set_xlabel("Architecture", fontsize=12)
    ax_rmse.set_ylabel("Average RMSE (N·m)", fontsize=12)
    ax_rmse.set_xlim(-0.6, n - 0.4)
    panel_label(ax_rmse, "a")

    proxy_train = Patch(facecolor="#888888", alpha=0.90, label="Train RMSE")
    proxy_test  = Patch(facecolor="#888888", alpha=0.60, hatch="////", label="Test RMSE")
    ax_rmse.legend(handles=[proxy_train, proxy_test], fontsize=9,
                   loc="upper left", bbox_to_anchor=(0.01, 0.99))

    # Panel (b): Test R2
    b_r2 = ax_r2.bar(x, test_r2, 0.50, color=bar_colors, alpha=0.82,
                     edgecolor="white", linewidth=0.8)
    for b, v in zip(b_r2, test_r2):
        if v == v and np.isfinite(v):
            ax_r2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                       f"{v:.4f}", ha="center", va="bottom", fontsize=9, rotation=75)

    valid_r2 = [v for v in test_r2 if v == v and np.isfinite(v)]
    if valid_r2:
        ax_r2.set_ylim(max(0.0, min(valid_r2) - 0.02), min(1.03, max(valid_r2) + 0.02))

    ax_r2.axhline(1.0, color="#888888", lw=1.0, ls="--", alpha=0.5)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(labels, rotation=0, ha="center", fontsize=11, fontweight="bold")
    ax_r2.set_xlabel("Architecture", fontsize=12)
    ax_r2.set_ylabel("Test R²", fontsize=12)
    ax_r2.set_xlim(-0.6, n - 0.4)
    panel_label(ax_r2, "b")

    # Panel (c): Test RMSE only -- zoomed y-axis
    b_te_solo = ax_z.bar(x, te_rmse, 0.55, color=bar_colors, alpha=0.88,
                         edgecolor="white", linewidth=0.8)
    for b, v in zip(b_te_solo, te_rmse):
        if v == v and np.isfinite(v):
            ax_z.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0001,
                      f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    tmin, tmax = zoom_ylim_1d(te_rmse, min_pad=0.0008, pad_rel=0.4)
    ax_z.set_ylim(tmin, tmax)
    fr = [v for v in te_rmse if v == v and np.isfinite(v)]
    range_txt = f"Test RMSE full range: [{min(fr):.4f}, {max(fr):.4f}] N·m" if fr else ""
    ax_z.set_xticks(x)
    ax_z.set_xticklabels(labels, rotation=0, ha="center", fontsize=11, fontweight="bold")
    ax_z.set_xlabel("Architecture", fontsize=12)
    ax_z.set_ylabel("Average RMSE (N·m) zoomed", fontsize=12)
    ax_z.set_xlim(-0.6, n - 0.4)
    ax_z.set_title("Best-per-type: magnified test RMSE", fontsize=12, fontweight="bold", pad=6)
    if range_txt:
        ax_z.text(0.99, 0.04, range_txt, transform=ax_z.transAxes, fontsize=8,
                  ha="right", va="bottom", color="#444444", style="italic")
    panel_label(ax_z, "c")
    ax_z.grid(True, axis="y", alpha=0.4)

    # Panel (d): Improvement vs best Black-Box test RMSE (mN·m)
    bb = best_blackbox_record(groups)
    if bb is not None:
        rmse_ref = rmse_scalar(bb, "test")
        delta_mnm = [((rmse_ref - v) * 1000.0) if (v == v and np.isfinite(v)) else float("nan")
                     for v in te_rmse]
        colors_d = [("#2b8a3e" if (d == d and d > 0) else "#c92a2a" if (d == d and d < 0) else "#666666")
                    for d in delta_mnm]
    else:
        delta_mnm = [float("nan")] * n
        colors_d = ["#666666"] * n
    b_d = ax_d.bar(x, delta_mnm, 0.55, color=colors_d, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax_d.axhline(0.0, color="#333333", lw=1.0)
    finite_d = [float(d) for d in delta_mnm if d == d and np.isfinite(d)]
    if finite_d:
        lo, hi = min(finite_d), max(finite_d)
        pad = max(0.4, 0.12 * (hi - lo + 1e-9))
        ax_d.set_ylim(lo - pad, hi + pad)
    for b, d in zip(b_d, delta_mnm):
        if d == d and np.isfinite(d):
            y0 = float(b.get_height())
            ylim = ax_d.get_ylim()
            h = ylim[1] - ylim[0]
            off = 0.03 * h * (1 if y0 >= 0 else -1)
            ax_d.text(
                b.get_x() + b.get_width() / 2, y0 + off, f"{d:+.1f}",
                ha="center", va="bottom" if y0 >= 0 else "top", fontsize=8, fontweight="bold",
            )
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, rotation=0, ha="center", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Architecture", fontsize=12)
    ax_d.set_ylabel("Delta Average RMSE vs Black-Box (mN·m)\n+ = better than Black-Box", fontsize=10)
    ax_d.set_xlim(-0.6, n - 0.4)
    panel_label(ax_d, "d")
    if bb is None:
        ax_d.text(0.5, 0.5, "No BlackBoxFNN in grid --\nbaseline for delta not defined.",
                  ha="center", va="center", transform=ax_d.transAxes, fontsize=10, color="#666666")
    else:
        ax_d.grid(True, axis="y", alpha=0.4)

    proxy_tr2 = Patch(facecolor="#888888", alpha=0.90, label="Train RMSE")
    proxy_te2 = Patch(facecolor="#888888", alpha=0.60, hatch="////", label="Test RMSE")
    arch_handles = [Patch(facecolor=type_colors[t], label=arch_short_label(t))
                    for t in sorted(type_colors)]
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.legend(handles=arch_handles + [proxy_tr2, proxy_te2],
               loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=len(arch_handles) + 2, fontsize=9)

    save_fig(fig, output_dir / "fig2_rmse_comparison.png")
