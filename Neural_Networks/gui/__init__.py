"""
Neural_Networks.gui
====================
Graphical interface components.

Sub-modules
-----------
preprocess_app   — Tkinter/matplotlib preprocessing GUI
visualizer_app   — Tkinter/matplotlib model comparison / visualizer GUI

These modules have heavy GUI dependencies (Tkinter, TkAgg matplotlib backend)
and must only be imported when a display is available.  They are isolated here
so the training pipeline (tui/, core/, cli/) can run headlessly without any
GUI imports.

Sub-module imports are intentionally deferred to avoid importing Tkinter and
matplotlib TkAgg backend at package import time (which would fail in headless
environments).
"""
