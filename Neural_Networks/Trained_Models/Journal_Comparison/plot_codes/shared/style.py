"""Global Matplotlib style + structural enforcement of the user's rules.

Rules (hard requirements):
  * SciencePlots base style (``pip install SciencePlots``) — applied, not optional.
  * Times New Roman everywhere (STIX for math: Times-metric δ/‖·‖ glyphs).
  * NO bold anywhere (every weight forced to ``normal``).
  * NO axes/figure titles.
  * NO ``(a)``/``(b)`` panel labels.
  * One horizontal frameless legend strip above the axes.
  * Large, legible fonts; everything driven by :class:`PlotConfig`.

The two ``assert_*`` helpers are invoked by :func:`figio.save_pdf` so the
constraints are enforced mechanically rather than by discipline.
"""

from __future__ import annotations

import logging
import re

import matplotlib

matplotlib.use("Agg")  # headless; PDF only
import matplotlib.pyplot as plt  # noqa: E402

from .config import PlotConfig  # noqa: E402

logger = logging.getLogger(__name__)

_PANEL_RE = re.compile(r"^\s*\(?\s*[a-hA-H]\s*\)?\s*$")


def apply_style(cfg: PlotConfig) -> None:
    """Apply the SciencePlots + non-bold Times serif theme from ``cfg``.

    SciencePlots is a hard requirement; if it is not installed we raise with an
    actionable message rather than silently rendering a different theme.
    """
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "SciencePlots is required for this figure suite. Install it with "
            "`pip install SciencePlots` (or `conda install -c conda-forge "
            "scienceplots`) and re-run."
        ) from exc

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": cfg.font_family,
        "font.serif": list(cfg.font_serif),
        "mathtext.fontset": cfg.mathtext_fontset,
        # --- no bold anywhere (every weight pinned to normal) ---
        "font.weight": "normal",
        "figure.titleweight": "normal",
        "axes.titleweight": "normal",

        "figure.figsize": (cfg.fig_w, cfg.fig_h),
        "figure.dpi": cfg.dpi_screen,
        "savefig.dpi": cfg.dpi_save,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": False,

        "axes.titlesize": 0.1,           # titles are banned anyway
        "axes.labelsize": cfg.axes_label_size,
        "axes.labelweight": cfg.axes_label_weight,
        "axes.linewidth": cfg.axes_linewidth,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=list(cfg.arch_colors.values())),

        "xtick.labelsize": cfg.tick_label_size,
        "ytick.labelsize": cfg.tick_label_size,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.direction": "in",
        "ytick.direction": "in",

        "grid.alpha": cfg.grid_alpha,
        "grid.linewidth": cfg.grid_linewidth,
        "grid.color": "lightgray",

        "legend.fontsize": cfg.legend_size,
        "legend.frameon": cfg.legend_frameon,
        "legend.framealpha": 0.0 if not cfg.legend_frameon else 0.95,
        "legend.edgecolor": "lightgray",
        "legend.fancybox": False,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.4,
        "legend.handletextpad": 0.5,

        "lines.linewidth": cfg.line_w,
        "lines.markersize": cfg.marker_size,
    })


def assert_no_title(fig: "plt.Figure") -> None:
    if getattr(fig, "_suptitle", None) is not None and fig._suptitle.get_text().strip():
        raise AssertionError("figure has a suptitle; titles are forbidden")
    for ax in fig.axes:
        for getter in (ax.get_title, lambda: ax.get_title(loc="left"),
                       lambda: ax.get_title(loc="right")):
            if getter() and getter().strip():
                raise AssertionError(f"axes has a title {getter()!r}; titles are forbidden")


def assert_no_panel_label(fig: "plt.Figure") -> None:
    """Reject lone ``(a)`` / ``b`` style annotations placed at axes corners."""
    for ax in fig.axes:
        for txt in ax.texts:
            s = txt.get_text()
            if s and _PANEL_RE.match(s):
                raise AssertionError(
                    f"text {s!r} looks like a panel label; panel labels are forbidden"
                )
