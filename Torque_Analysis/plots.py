"""
All plotting functions — each produces one figure.
plots.py

Merged from plots.py + plots_bulk.py:
  - set_headless(True) for batch mode (Agg backend, plt.close)
  - set_headless(False) for interactive mode (plt.show)
  - All functions accept optional `title` and `save_path`
  - Handles both 5-joint and 6-joint views
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import config as C

# ============================================================
# Mode control
# ============================================================
_HEADLESS = False


def set_headless(headless: bool = True):
    """
    Switch between interactive and headless (batch) mode.

    Call BEFORE any plotting — must be set before matplotlib
    creates its first figure in some backends.
    """
    global _HEADLESS
    if headless:
        matplotlib.use("Agg")
    _HEADLESS = headless


def _finish(fig, save_path: str = None):
    """Save and/or show figure depending on mode."""
    if save_path:
        fig.savefig(save_path, dpi=C.DPI, bbox_inches="tight")
    if _HEADLESS:
        plt.close(fig)
    else:
        plt.show()


# ==================================================================
# 1.  Single-joint torque comparison
# ==================================================================
def plot_single_joint(t, tau_load, tau_urdf,
                      joint: int = 0, torque_urdf_fric=None,
                      title=None, save_path=None):
    """Load register vs URDF RNEA (± friction) for one joint."""
    fig, ax = plt.subplots(figsize=(C.FIG_WIDTH, C.FIG_HEIGHT), dpi=C.DPI)

    ax.plot(t, tau_load[:, joint],
            label="Load Register (measured)", lw=1.0, color="C0")
    ax.plot(t, tau_urdf[:, joint],
            label="URDF RNEA", lw=1.2, color="C2")

    if torque_urdf_fric is not None:
        ax.plot(t, torque_urdf_fric[:, joint],
                label="URDF RNEA + Friction", lw=1.2, ls="--", color="C3")

    ax.axhline(0, color='k', lw=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title(title or f"Joint {joint + 1} — Torque Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _finish(fig, save_path)


# ==================================================================
# 1b. Single joint — load only (for joints without RNEA model)
# ==================================================================
def plot_load_only(t, tau_load, joint: int = 5,
                   title=None, save_path=None):
    """Load register only (for passive joints without RNEA)."""
    fig, ax = plt.subplots(figsize=(C.FIG_WIDTH, C.FIG_HEIGHT), dpi=C.DPI)

    ax.plot(t, tau_load[:, joint],
            label="Load Register", lw=1.0, color="C0")

    ax.axhline(0, color='k', lw=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title(title or f"Joint {joint + 1} — Load Only")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.95, "No RNEA model for this joint",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="gray", style="italic")
    plt.tight_layout()
    _finish(fig, save_path)


# ==================================================================
# 2.  All joints — 3×2 grid
# ==================================================================
def plot_all_joints(t, tau_load, tau_urdf,
                    nq: int, dof: int = 5,
                    torque_urdf_fric=None,
                    title=None, save_path=None):
    """3×2 subplot grid, one panel per joint."""
    fig, axes = plt.subplots(3, 2, figsize=(C.FIG_WIDTH, 12),
                             sharex=True, dpi=C.DPI)

    for j, ax in enumerate(axes.flat):
        if j >= dof:
            ax.set_visible(False)
            continue

        ax.plot(t, tau_load[:, j],
                label="Load Reg (measured)", lw=1.0, color="C0")

        if j < nq:
            ax.plot(t, tau_urdf[:, j],
                    label="URDF RNEA", lw=1.2, color="C2")
            if torque_urdf_fric is not None:
                ax.plot(t, torque_urdf_fric[:, j],
                        label="URDF + Friction", lw=1.0, ls="--", color="C3")
        else:
            ax.text(0.98, 0.95, "No RNEA model",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="gray", style="italic")

        ax.axhline(0, color='k', lw=0.3)

        label = f"Joint {j + 1}"
        if j == 0:
            label += " (yaw)"
        elif j == 5:
            label += " (tool)"
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("N·m", fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(title or "All Joints — Torque Comparison",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _finish(fig, save_path)


# ==================================================================
# 3.  Joint angles over time
# ==================================================================
def plot_joint_angles(t, q: np.ndarray, nq: int,
                      title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(C.FIG_WIDTH, C.FIG_HEIGHT), dpi=C.DPI)
    for j in range(nq):
        ax.plot(t, np.degrees(q[:, j]), label=f"Joint {j + 1}", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title(title or "Joint Angles")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _finish(fig, save_path)


# ==================================================================
# 4.  Velocity & acceleration profiles
# ==================================================================
def plot_vel_acc(t, qd: np.ndarray, qdd: np.ndarray,
                 nq: int, title=None, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(C.FIG_WIDTH, 8),
                                   sharex=True, dpi=C.DPI)
    for j in range(nq):
        ax1.plot(t, qd[:, j],  label=f"J{j + 1}", lw=0.7)
        ax2.plot(t, qdd[:, j], label=f"J{j + 1}", lw=0.7)

    ax1.set_ylabel("Velocity (rad/s)")
    ax1.set_title(title or "Joint Velocities (smoothed)")
    ax1.legend(fontsize=7, ncol=nq)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acceleration (rad/s²)")
    ax2.set_title("Joint Accelerations (smoothed)")
    ax2.legend(fontsize=7, ncol=nq)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _finish(fig, save_path)


# ==================================================================
# 5.  Raw servo signals (load, voltage)
# ==================================================================
def plot_raw_signals(t, L: dict, dof: int = 5,
                     title=None, save_path=None):
    """Two-row plot: load register, bus voltage."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(C.FIG_WIDTH, 7),
                                   sharex=True, dpi=C.DPI)
    for j in range(dof):
        ax1.plot(t, L["load"][:, j],    label=f"J{j+1}", lw=0.5)
        ax2.plot(t, L["voltage"][:, j], label=f"J{j+1}", lw=0.5)

    ax1.set_ylabel("Load (raw)")
    ax1.set_title(title or "Raw Load Register")
    ax1.legend(fontsize=7, ncol=dof)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Voltage (V)")
    ax2.set_title("Bus Voltage")
    ax2.legend(fontsize=7, ncol=dof)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _finish(fig, save_path)


# ==================================================================
# 6.  End-effector trajectory (3D + projections)
# ==================================================================
def plot_ee_trajectory(L: dict, title=None, save_path=None):
    fig = plt.figure(figsize=(C.FIG_WIDTH, 6), dpi=C.DPI)

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(L["cmd_ee"][:, 0], L["cmd_ee"][:, 1], L["cmd_ee"][:, 2],
             label="Cmd", lw=0.8)
    ax1.plot(L["act_ee"][:, 0], L["act_ee"][:, 1], L["act_ee"][:, 2],
             label="Act", lw=0.8)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("EE Trajectory (3D)")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(122)
    ax2.plot(L["cmd_ee"][:, 0], L["cmd_ee"][:, 2], label="Cmd", lw=0.8)
    ax2.plot(L["act_ee"][:, 0], L["act_ee"][:, 2], label="Act", lw=0.8)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("EE Trajectory (XZ plane)")
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    _finish(fig, save_path)


# ==================================================================
# 7.  Tracking error
# ==================================================================
def plot_tracking_error(t, ee_err: np.ndarray,
                        title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(C.FIG_WIDTH, 4), dpi=C.DPI)
    ax.plot(t, ee_err * 1000, lw=0.8, color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EE Error (mm)")
    ax.set_title(title or "End-Effector Tracking Error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _finish(fig, save_path)


# ==================================================================
# 8.  Gravity-only torque vs full RNEA
# ==================================================================
def plot_gravity_vs_full(t, torque_urdf, torque_gravity,
                         nq: int, title=None, save_path=None):
    fig, axes = plt.subplots(3, 2, figsize=(C.FIG_WIDTH, 12),
                             sharex=True, dpi=C.DPI)
    for j, ax in enumerate(axes.flat):
        if j >= nq:
            ax.set_visible(False)
            continue
        ax.plot(t, torque_urdf[:, j],    label="Full RNEA", lw=0.8)
        ax.plot(t, torque_gravity[:, j], label="Gravity only", lw=1.0, ls="--")
        ax.axhline(0, color='k', lw=0.3)
        ax.set_title(f"Joint {j + 1}", fontsize=10)
        ax.set_ylabel("N·m", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(title or "Full RNEA vs Gravity-Only Torque",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _finish(fig, save_path)


# ==================================================================
# 9.  Residual plot (Load - URDF = unmodeled dynamics)
# ==================================================================
def plot_residual(t, tau_load, tau_urdf, tau_urdf_fric=None,
                  nq: int = 5, nq_model: int = None,
                  title=None, save_path=None):
    """Shows the residual (what remains after analytical model)."""
    if nq_model is None:
        nq_model = nq

    fig, axes = plt.subplots(3, 2, figsize=(C.FIG_WIDTH, 12),
                             sharex=True, dpi=C.DPI)

    for j, ax in enumerate(axes.flat):
        if j >= nq:
            ax.set_visible(False)
            continue

        if j < nq_model:
            resid_raw = tau_load[:, j] - tau_urdf[:, j]
            ax.plot(t, resid_raw, label="Load − RNEA", lw=0.7, color="C0")

            if tau_urdf_fric is not None:
                resid_fric = tau_load[:, j] - tau_urdf_fric[:, j]
                ax.plot(t, resid_fric,
                        label="Load − (RNEA+Fric)", lw=0.7, color="C3")
        else:
            # No RNEA → residual = full load
            ax.plot(t, tau_load[:, j],
                    label="Load (no model)", lw=0.7, color="C0")

        ax.axhline(0, color='k', lw=0.3)

        label = f"Joint {j + 1}"
        if j == 0:
            label += " (yaw)"
        elif j == 5:
            label += " (tool)"
        ax.set_title(f"{label} — Residual", fontsize=10)
        ax.set_ylabel("N·m", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(
        title or "Torque Residual (unmodeled dynamics)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    _finish(fig, save_path)
