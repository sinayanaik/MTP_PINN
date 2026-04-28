"""Fig 3 — R2 comparison (train vs test), single panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..io.records import arch_short_label, best_per_type, split_scalar
from ..style import type_color_map
from ._common import save_fig


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    all_recs = best_per_type(groups)
    if not all_recs:
        return

    type_colors = type_color_map(list(groups.keys()))
    bar_colors = [type_colors.get(r.get("model_type", "?"), "gray") for r in all_recs]
    labels = [arch_short_label(r.get("model_type", "?")) for r in all_recs]

    n = len(all_recs)
    x = np.arange(n)
    bw = 0.32

    tv  = [split_scalar(r, "test",  "r2_overall") for r in all_recs]
    trv = [split_scalar(r, "train", "r2_overall") for r in all_recs]

    fig, ax = plt.subplots(1, 1, figsize=(max(7, n * 2.4), 6.0))

    b_tr = ax.bar(x - bw / 2, trv, bw, color=bar_colors, alpha=0.90,
                  edgecolor="white", linewidth=1.0)
    b_te = ax.bar(x + bw / 2, tv,  bw, color=bar_colors, alpha=0.60,
                  edgecolor="white", linewidth=1.0, hatch="////")

    for b, v in zip(b_tr, trv):
        if v == v and np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0008,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                    rotation=75)
    for b, v in zip(b_te, tv):
        if v == v and np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0008,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                    rotation=75)

    valid = [v for v in trv + tv if v == v and np.isfinite(v)]
    lo = max(0.0, min(valid) - 0.03) if valid else 0.7
    hi = min(1.03, max(valid) + 0.03) if valid else 1.03
    ax.set_ylim(lo, hi)
    ax.axhline(1.0, color="dimgray", lw=1.0, alpha=0.4, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Architecture", fontsize=13, fontweight="bold")
    ax.set_ylabel("R²", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(True, axis="y", alpha=0.35)

    proxy_tr = Patch(facecolor="gray", alpha=0.90, label="Train")
    proxy_te = Patch(facecolor="gray", alpha=0.60, hatch="////", label="Test")
    arch_handles = [Patch(facecolor=type_colors[t], label=arch_short_label(t))
                    for t in sorted(type_colors)]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(
        handles=arch_handles + [proxy_tr, proxy_te],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(arch_handles) + 2,
        fontsize=11,
    )

    save_fig(fig, output_dir / "fig3_r2_comparison.pdf")

