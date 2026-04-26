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
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
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
    ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left")
