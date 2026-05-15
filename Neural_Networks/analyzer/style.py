"""Plot styling: Times New Roman, scienceplots, and consistent layouts."""

from __future__ import annotations

import logging
from typing import Iterable

import matplotlib.pyplot as plt

from .config import _ARCH_COLOR_ORDER

logger = logging.getLogger(__name__)


def setup_plot_style(palette: str = "tab10") -> None:
    """Apply the global style with Times New Roman and consistent grid/ticks."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science', 'no-latex'])
    except ImportError:
        logger.warning("scienceplots not found, using default style.")

    plt.rcParams.update({
        # Typography: Use STIX fonts for a Times New Roman look with LaTeX support
        "text.usetex": False, # Use STIX instead of system TeX for faster/more portable rendering
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "serif"],
        "mathtext.fontset": "stix",
        
        # Figure and Layout
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": False,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",

        # Axes, Titles, and Labels
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.5,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,

        # Ticks and Grid
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,
        "grid.color": "lightgray",

        # Legend
        "legend.fontsize": 11,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "lightgray",
        "legend.fancybox": False,
        "legend.loc": "upper center", # Default to top as requested
        
        # Lines and markers
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })


def type_color_map(model_types: Iterable[str]) -> dict[str, tuple]:
    """Architecture -> colour with stable ordering across all figures."""
    import matplotlib.pyplot as _plt
    seen = set(model_types)
    ordered = [t for t in _ARCH_COLOR_ORDER if t in seen]
    extras = sorted(t for t in seen if t not in _ARCH_COLOR_ORDER)
    cmap = _plt.get_cmap("tab10")
    return {t: cmap(i / 10.0) for i, t in enumerate(ordered + extras)}


def panel_label(ax: "plt.Axes", letter: str, fontsize: float = 14.0, y_offset: float = -0.22) -> None:
    """Standardized panel label (a), (b) etc. positioned at bottom middle."""
    ax.text(0.5, y_offset, f"({letter})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="center")
