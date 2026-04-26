"""Grid-search trained-model analyzer.

Public entrypoints:
    from Neural_Networks.analyzer import run_all, main
    from Neural_Networks.analyzer.plots import PLOTS
"""
from __future__ import annotations

from .cli import main
from .compute.enrich import enrich_records
from .io.scan import group_by_model_type, scan_trained_models
from .plots import PLOTS
from .runner import run_all

__all__ = [
    "run_all",
    "main",
    "scan_trained_models",
    "group_by_model_type",
    "enrich_records",
    "PLOTS",
]
