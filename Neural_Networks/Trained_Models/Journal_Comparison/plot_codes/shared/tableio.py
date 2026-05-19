"""CSV table writer — the table-side mirror of :func:`figio.save_pdf`.

Used only where a plot would be trivially simple (single value per
architecture); everything richer stays a figure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PlotConfig


def save_table(df: pd.DataFrame, name: str, cfg: PlotConfig) -> Path:
    """Write ``df`` to ``<tables_dir>/<name>.csv`` (4-decimal floats)."""
    out_dir = Path(cfg.tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(name).stem}.csv"
    df.to_csv(path, index=False, float_format=cfg.table_float_fmt)
    return path
