"""Fig 1 — train/val loss curves, best-per-type."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..io.records import arch_short_label, best_per_type
from ..style import type_color_map
from ._common import save_fig


def plot(groups: dict[str, list[dict[str, Any]]], output_dir: Path, **_: Any) -> None:
    all_recs = best_per_type(groups)
    n = len(all_recs)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False,
    )
    axes_flat = axes.flatten()
    type_colors = type_color_map(list(groups.keys()))

    panel_letters = "abcdefgh"
    all_losses: list[float] = []

    for idx, rec in enumerate(all_recs):
        ax = axes_flat[idx]
        hist = rec.get("_history", {})
        mtype = rec.get("model_type", "unknown")
        best_ep = rec.get("_best_epoch", -1)

        tl = hist.get("train_loss", [])
        vl = hist.get("val_loss",   [])
        ep = hist.get("epoch", list(range(1, max(len(tl), len(vl), 1) + 1)))

        c = type_colors.get(mtype, "steelblue")

        if tl:
            ax.plot(ep[:len(tl)], tl, color=c, lw=2.0, label="Train Loss")
            all_losses.extend(tl)
        if vl:
            ax.plot(ep[:len(vl)], vl, color=c, lw=2.0, ls="--", alpha=0.8, label="Val Loss")
            all_losses.extend(vl)
        if best_ep > 0:
            ax.axvline(best_ep, color="#CC0000", lw=1.2, ls=":", alpha=0.85,
                       label=f"Best checkpoint (ep {best_ep})")

        letter = panel_letters[idx] if idx < len(panel_letters) else str(idx)
        ax.set_title(f"({letter}) {arch_short_label(mtype)}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("MSE Loss", fontsize=12)
        ax.grid(True, alpha=0.25)

    finite_losses = [v for v in all_losses if np.isfinite(v)]
    if finite_losses:
        lo = max(0.0, np.percentile(finite_losses, 1) * 0.9)
        hi = np.percentile(finite_losses, 99) * 1.05
        for idx in range(n):
            axes_flat[idx].set_ylim(lo, hi)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()),
                   loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=len(seen), fontsize=11, framealpha=0.95,
                   edgecolor="#aaaaaa", borderpad=0.8)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    save_fig(fig, output_dir / "fig1_training_dynamics.png")
