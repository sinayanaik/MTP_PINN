"""
Neural_Networks.analysis.plots
================================
Matplotlib-based comparison plots read from models_registry.yaml.

All analysis logic is maintained in the parent module
``Neural_Networks.models_analysis`` for backward compatibility.
This module is a thin re-export shim so analysis can be launched from
the canonical sub-package path.

Usage
-----
    python -m Neural_Networks.models_analysis    # backward-compatible entry point
    python -m Neural_Networks.analysis.plots     # sub-package entry point (equivalent)
"""

from Neural_Networks.analysis.registry import main  # noqa: F401

if __name__ == "__main__":
    main()
