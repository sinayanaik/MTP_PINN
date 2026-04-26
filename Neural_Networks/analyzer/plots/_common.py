"""Helpers shared across plot modules."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def save_fig(fig: "plt.Figure", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")
    fig.savefig(str(pdf_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    logger.info("Saved: %s  +  %s", pdf_path, png_path)
    plt.draw()


def fmt(v: float, decimals: int = 5) -> str:
    return f"{v:.{decimals}f}" if v == v else "   -   "


def annotate_bars(ax, bars, vals: list[float],
                  rotation: int = 90, fontsize: float = 8.0) -> None:
    for bar, v in zip(bars, vals):
        if v == v:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0005,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=fontsize, rotation=rotation)


def zoom_ylim_1d(
    values: list[float], *, min_pad: float, pad_rel: float = 0.35,
) -> tuple[float, float]:
    vv = [float(v) for v in values if v == v and np.isfinite(v)]
    if not vv:
        return 0.0, 1.0
    lo, hi = min(vv), max(vv)
    span = max(hi - lo, 1e-9)
    pad = max(min_pad, pad_rel * span)
    return lo - pad, hi + pad


def estimate_params(hidden_layers: list[int], n_in: int = 15, n_out: int = 5) -> int:
    sizes = [n_in] + list(hidden_layers) + [n_out]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


def hp_val_str(v: Any) -> str:
    if isinstance(v, list):
        return "x".join(str(x) for x in v)
    if isinstance(v, float):
        if abs(v) < 0.01 or abs(v) > 999:
            s = f"{v:.2e}"
            s = re.sub(r"0\.00e\+00", "0", s)
            return s
        return f"{v:.4g}"
    s = str(v)
    s = re.sub(r"^0e\+00$", "0", s)
    return s
