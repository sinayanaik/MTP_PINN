"""PDF-only figure writer with mechanical constraint enforcement."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .config import PlotConfig
from .style import assert_no_panel_label, assert_no_title


def save_pdf(fig: "plt.Figure", name: str, cfg: PlotConfig) -> Path:
    """Save ``fig`` as ``<figures_dir>/<name>.pdf`` (300 dpi, tight) and close it.

    Enforces the no-title / no-panel-label rules before writing.
    """
    if cfg.enforce_no_title:
        assert_no_title(fig)
    if cfg.enforce_no_panel_label:
        assert_no_panel_label(fig)

    out_dir = Path(cfg.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(name).stem}.pdf"
    fig.savefig(str(path), dpi=cfg.dpi_save, bbox_inches="tight")
    plt.close(fig)
    return path
