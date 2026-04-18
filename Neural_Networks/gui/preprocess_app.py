"""
Neural_Networks.gui.preprocess_app
====================================
Tkinter/matplotlib interactive preprocessing GUI.

All preprocessing GUI logic is maintained in the parent module
``Neural_Networks.preprocess_data`` for backward compatibility.
This module is a thin re-export shim so the GUI can also be launched
from the canonical sub-package path.

Usage
-----
    python -m Neural_Networks.preprocess_data    # backward-compatible entry point
    python -m Neural_Networks.gui.preprocess_app # sub-package entry point (equivalent)
"""

from Neural_Networks.apps.preprocess import main  # noqa: F401

if __name__ == "__main__":
    main()
