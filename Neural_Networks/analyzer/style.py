"""Plot styling: scienceplots theme + architecture-to-colour mapping."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt

from .config import _ARCH_COLOR_ORDER


def setup_plot_style(palette: str = "tab10") -> None:
    """Apply the science journal style. Call once before any plotting."""
    import scienceplots  # noqa: F401 — registers 'science' style
    plt.style.use(['science', 'no-latex'])
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": False,
        # Title / label sizes and weights
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        # Axis frame
        "axes.linewidth": 1.5,
        # Tick sizes and widths
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        # Lines and markers
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        # Legend
        "legend.fontsize": 11,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "lightgray",
        # Grid
        "grid.alpha": 0.35,
        "grid.linewidth": 0.8,
    })


def type_color_map(model_types: Iterable[str]) -> dict[str, tuple]:
    """Architecture -> colour with stable ordering across all figures."""
    import matplotlib.pyplot as _plt
    seen = set(model_types)
    ordered = [t for t in _ARCH_COLOR_ORDER if t in seen]
    extras = sorted(t for t in seen if t not in _ARCH_COLOR_ORDER)
    cmap = _plt.get_cmap("tab10")
    return {t: cmap(i / 9.0) for i, t in enumerate(ordered + extras)}


def panel_label(ax: "plt.Axes", letter: str, fontsize: float = 13.0) -> None:
    ax.text(0.02, 0.03, f"({letter})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="bottom", ha="left")
