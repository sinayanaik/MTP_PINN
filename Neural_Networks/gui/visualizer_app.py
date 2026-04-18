"""
Neural_Networks.gui.visualizer_app
=====================================
Tkinter/matplotlib model comparison visualizer.

All visualizer GUI logic is maintained in the parent module
``Neural_Networks.visualizer`` for backward compatibility.
This module is a thin re-export shim so the GUI can also be launched
from the canonical sub-package path.

Usage
-----
    python -m Neural_Networks.visualizer          # backward-compatible entry point
    python -m Neural_Networks.gui.visualizer_app  # sub-package entry point (equivalent)
"""

from Neural_Networks.apps.visualizer import main  # noqa: F401

if __name__ == "__main__":
    main()
