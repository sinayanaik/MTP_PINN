"""
Headless plot functions for bulk analysis.
plots_bulk.py

This file is now a thin backwards-compatibility wrapper.
All plotting logic lives in plots.py with set_headless(True).

Existing code that imports from plots_bulk will continue to work.
For new code, use:
    from Torque_Analysis.plots import set_headless, plot_all_joints, ...
    set_headless(True)
"""

from .plots import (
    set_headless,
    plot_all_joints,
    plot_residual,
    plot_gravity_vs_full,
    plot_single_joint,
    plot_load_only,
    plot_joint_angles,
    plot_vel_acc,
    plot_raw_signals,
    plot_ee_trajectory,
    plot_tracking_error,
)

# Auto-enable headless mode when this module is imported
set_headless(True)

__all__ = [
    "set_headless",
    "plot_all_joints",
    "plot_residual",
    "plot_gravity_vs_full",
    "plot_single_joint",
    "plot_load_only",
    "plot_joint_angles",
    "plot_vel_acc",
    "plot_raw_signals",
    "plot_ee_trajectory",
    "plot_tracking_error",
]
