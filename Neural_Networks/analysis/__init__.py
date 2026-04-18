"""
Neural_Networks.analysis
=========================
Post-training model analysis and visualization utilities.

Sub-modules
-----------
plots   — matplotlib-based registry plots (RMSE comparison, family bars, etc.)

This sub-package contains read-only analysis tools that operate on a trained
models_registry.yaml and saved artefacts.  It has no dependency on the Rich
TUI, the training pipeline, or the interactive CLI.
"""

from Neural_Networks.analysis.plots import main  # noqa: F401

__all__ = ["main"]
