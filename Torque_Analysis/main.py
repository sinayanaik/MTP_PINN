#!/usr/bin/env python3
"""
Master entry point for torque analysis.
main.py

Improvements:
  - Removed dummy_current hack (diagnostics no longer needs it)
  - Added logging setup
  - Cleaner diagnostics call
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from Torque_Analysis import config as C
from Torque_Analysis.data_loader import load_log, print_summary
from Torque_Analysis.utils import ticks_to_radians
from Torque_Analysis.torque import (
    torque_from_load,
    build_pinocchio_model,
    torque_from_urdf,
    torque_gravity_only,
    torque_friction,
)
from Torque_Analysis.diagnostics import run_all as run_diagnostics
from Torque_Analysis import plots


def main():
    C.setup_logging()

    # ========================================================
    # 1. LOAD DATA
    # ========================================================
    print("Loading data...")
    L, M, N = load_log(C.LOG_JSON)
    print_summary(L, M, N)

    AJ = C.ACTIVE_JOINTS  # 5 active joints (joint 6 = passive tool)

    # ========================================================
    # 2. JOINT ANGLES → RADIANS
    # ========================================================
    act_rad = ticks_to_radians(
        L["act_pos"], M["joint_map"], M["ticks_to_rad"], C.DOF
    )

    # ========================================================
    # 3. BUILD PINOCCHIO MODEL
    # ========================================================
    print(f"\nBuilding Pinocchio model (MASS_SCALE={C.MASS_SCALE})...")
    model, pdata, nq = build_pinocchio_model(
        C.XACRO_PATH,
        mass_scale=C.MASS_SCALE,
        extra_masses=C.EXTRA_MASSES,
    )
    q = act_rad[:, :nq]
    total_mass = sum(model.inertias[i].mass for i in range(model.njoints))
    print(f"  nq={nq}  total_mass={total_mass:.4f} kg")

    # ========================================================
    # 4. COMPUTE TORQUES
    # ========================================================
    print("Computing torques...")

    # (a) Load-register — direction-corrected (measured reference)
    tau_load = torque_from_load(
        L["load"], L["voltage"], joint_map=M["joint_map"]
    )

    # (b) URDF inverse dynamics
    tau_urdf, qd, qdd = torque_from_urdf(model, pdata, q, L["t"])

    # (c) Gravity-only
    tau_gravity = torque_gravity_only(model, pdata, q)

    # (d) Friction (smooth tanh)
    tau_fric = torque_friction(qd)
    tau_urdf_fric = tau_urdf + tau_fric

    # (e) Residual = unmodeled dynamics
    tau_residual = tau_load[:, :nq] - tau_urdf_fric[:, :nq]

    print("Done.\n")

    # ========================================================
    # 5. SANITY CHECK
    # ========================================================
    print("=" * 70)
    print(f"SANITY CHECK — Active joints 0..{AJ-1}")
    print("=" * 70)
    print(f"  {'Jnt':>3}  {'Load RMS':>10}  {'RNEA RMS':>10}  "
          f"{'RNEA+F RMS':>10}  {'Resid RMS':>10}  {'RNEA/Load':>10}")

    for j in range(AJ):
        rms_l = np.sqrt(np.mean(tau_load[:, j] ** 2))
        rms_u = np.sqrt(np.mean(tau_urdf[:, j] ** 2))
        rms_f = np.sqrt(np.mean(tau_urdf_fric[:, j] ** 2))
        rms_r = np.sqrt(np.mean(tau_residual[:, j] ** 2)) if j < nq else float('nan')
        ratio = rms_u / rms_l if rms_l > 1e-6 else float('nan')
        print(f"  {j:>3d}  {rms_l:>10.4f}  {rms_u:>10.4f}  "
              f"{rms_f:>10.4f}  {rms_r:>10.4f}  {ratio:>10.2f}")

    # ========================================================
    # 6. DIAGNOSTICS
    # ========================================================
    run_diagnostics(
        L, M, model, pdata,
        q, qd, qdd,
        tau_load, tau_urdf,
    )

    # ========================================================
    # 7. PLOTS
    # ========================================================
    print("\n\nGenerating plots...\n")

    # 7a — All active joints: Load vs URDF vs URDF+Friction
    plots.plot_all_joints(
        L["t"], tau_load, tau_urdf,
        nq=min(nq, AJ), dof=AJ,
        torque_urdf_fric=tau_urdf_fric,
    )

    # 7b — Residual
    plots.plot_residual(
        L["t"], tau_load, tau_urdf,
        tau_urdf_fric=tau_urdf_fric, nq=AJ,
    )

    # 7c — Gravity vs full RNEA
    plots.plot_gravity_vs_full(L["t"], tau_urdf, tau_gravity, nq=AJ)

    # Uncomment as needed:
    # for j in [1, 2]:
    #     plots.plot_single_joint(
    #         L["t"], tau_load, tau_urdf,
    #         joint=j, torque_urdf_fric=tau_urdf_fric,
    #     )
    # plots.plot_joint_angles(L["t"], q[:, :AJ], AJ)
    # plots.plot_vel_acc(L["t"], qd[:, :AJ], qdd[:, :AJ], AJ)
    # plots.plot_raw_signals(L["t"], L, AJ)
    # plots.plot_ee_trajectory(L)
    # plots.plot_tracking_error(L["t"], L["ee_err"])


if __name__ == "__main__":
    main()
