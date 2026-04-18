"""
All diagnostic checks — run these FIRST to identify issues before trusting results.
diagnostics.py

Improvements:
  - Removed torque_current from run_all (no longer used in pipeline)
  - DIAG 8 and DIAG 9 simplified (current-based torque is deprecated)
  - Added logging alongside print for structured output
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def diag_pinocchio_joints(model, joint_map: list):
    """DIAG 1: Compare pinocchio joint tree vs log joint_map."""
    print("\n" + "=" * 65)
    print("DIAG 1 — Pinocchio joints vs Log joint_map")
    print("=" * 65)
    print(f"  nq={model.nq}  nv={model.nv}  njoints={model.njoints}")

    print("\n  Pinocchio joint tree:")
    for i in range(model.njoints):
        print(f"    pin[{i}]: '{model.names[i]}'  "
              f"type={model.joints[i].shortname()}")

    print("\n  Log joint_map:")
    for i, jm in enumerate(joint_map):
        name = jm.get("joint_name", jm.get("name", "?"))
        sid  = jm.get("servo_id", "?")
        print(f"    log[{i}]: servo_id={sid}  name={name}  "
              f"dir={jm['direction']}  center={jm['ticks_center']}")

    # Check ordering match
    pin_names = [model.names[i] for i in range(1, model.njoints)]  # skip 'universe'
    log_names = [jm.get("joint_name", jm.get("name", "?")) for jm in joint_map]

    if len(pin_names) <= len(log_names):
        match = all(p == l for p, l in zip(pin_names, log_names))
        status = "✓ YES" if match else "✗ NO — REORDER NEEDED"
        print(f"\n  Name ordering match (first {len(pin_names)}): {status}")
        if not match:
            logger.warning("Joint ordering mismatch between Pinocchio and log!")


def diag_urdf_masses(model):
    """DIAG 2: Check link masses and CoM positions."""
    print("\n" + "=" * 65)
    print("DIAG 2 — URDF link masses and inertias")
    print("=" * 65)

    total = 0.0
    for i in range(model.njoints):
        m = model.inertias[i].mass
        total += m
        com = model.inertias[i].lever
        inertia = model.inertias[i].inertia
        diag = [inertia[0, 0], inertia[1, 1], inertia[2, 2]]
        print(f"  joint {i} '{model.names[i]}':")
        print(f"    mass = {m:.6f} kg")
        print(f"    CoM  = [{com[0]:+.6f}, {com[1]:+.6f}, {com[2]:+.6f}]")
        print(f"    Ixx,Iyy,Izz = [{diag[0]:.2e}, {diag[1]:.2e}, {diag[2]:.2e}]")

    print(f"\n  >>> TOTAL MASS: {total:.4f} kg <<<")
    if total < 0.1:
        logger.warning("Total mass %.4f kg — likely placeholder values!", total)
        print("  *** WARNING: Total mass < 0.1 kg — likely placeholder values! ***")
    elif total < 0.5:
        logger.warning("Total mass %.4f kg — seems low for a 6-DOF arm", total)
        print("  *** WARNING: Total mass < 0.5 kg — seems low for a 6-DOF arm ***")


def diag_gravity_torque(model, data, q: np.ndarray):
    """DIAG 3: Gravity vector and gravity-only torque at midpoint."""
    import pinocchio as pin

    print("\n" + "=" * 65)
    print("DIAG 3 — Gravity vector & gravity-only torque")
    print("=" * 65)
    print(f"  model.gravity = {model.gravity}")

    mid = q.shape[0] // 2
    zero = np.zeros(model.nv)
    tau_g = pin.rnea(model, data, q[mid], zero, zero)

    print(f"  q_mid (deg):  {np.degrees(q[mid]).round(2)}")
    print(f"  tau_gravity:  {tau_g.round(6)}")
    print(f"  max|tau_g| =  {np.abs(tau_g).max():.6f} N·m")

    # Also at home pose
    q_home = np.zeros(model.nq)
    tau_home = pin.rnea(model, data, q_home, zero, zero)
    print(f"\n  At home (q=0):")
    print(f"  tau_gravity:  {tau_home.round(6)}")


def diag_joint_angles(q: np.ndarray, nq: int):
    """DIAG 4: Joint angle ranges."""
    print("\n" + "=" * 65)
    print("DIAG 4 — Joint angle ranges")
    print("=" * 65)
    for j in range(nq):
        lo, hi = q[:, j].min(), q[:, j].max()
        mean_q = q[:, j].mean()
        rng = np.degrees(hi - lo)
        print(f"  q[{j}]: [{np.degrees(lo):+8.2f}, {np.degrees(hi):+8.2f}] deg  "
              f"range={rng:6.2f} deg  "
              f"mean={np.degrees(mean_q):+8.2f} deg")
        if rng < 0.1:
            logger.warning("Joint %d range is only %.2f deg — possibly stuck", j, rng)


def diag_timestamps(t: np.ndarray):
    """DIAG 5: Timestamp consistency."""
    print("\n" + "=" * 65)
    print("DIAG 5 — Timestamps")
    print("=" * 65)

    dt = np.diff(t)
    n_bad = np.sum(dt <= 0)
    hz = 1 / dt.mean() if dt.mean() > 0 else 0

    print(f"  dt mean:  {dt.mean() * 1000:.2f} ms")
    print(f"  dt std:   {dt.std() * 1000:.2f} ms")
    print(f"  dt min:   {dt.min() * 1000:.2f} ms")
    print(f"  dt max:   {dt.max() * 1000:.2f} ms")
    print(f"  Any dt<=0? {n_bad > 0} ({n_bad} samples)")
    print(f"  Effective Hz: {hz:.1f}")

    # Flag outliers
    outliers = np.sum(dt > 3 * dt.mean())
    if outliers > 0:
        logger.warning("%d timesteps > 3× mean dt", outliers)
        print(f"  *** WARNING: {outliers} timesteps > 3× mean dt ***")


def diag_velocities(qd: np.ndarray, qdd: np.ndarray, nq: int):
    """DIAG 6: Velocity and acceleration magnitudes."""
    print("\n" + "=" * 65)
    print("DIAG 6 — Velocity / Acceleration magnitudes")
    print("=" * 65)
    for j in range(nq):
        max_vel = np.abs(qd[:, j]).max()
        max_acc = np.abs(qdd[:, j]).max()
        print(f"  joint {j}: max|qd| = {max_vel:8.3f} rad/s   "
              f"max|qdd| = {max_acc:8.3f} rad/s²")
        if max_vel > 50:
            logger.warning("Joint %d max velocity %.1f rad/s — possible blowup", j, max_vel)


def diag_raw_signals(L: dict, dof: int = 6):
    """DIAG 7: Raw load and voltage ranges."""
    print("\n" + "=" * 65)
    print("DIAG 7 — Raw load / voltage")
    print("=" * 65)

    for j in range(dof):
        ld = L["load"][:, j]
        v  = L["voltage"][:, j]
        print(f"  joint {j}:")
        print(f"    load    : [{ld.min():+7.1f}, {ld.max():+7.1f}]  "
              f"mean={ld.mean():+6.1f}")
        print(f"    voltage : [{v.min():6.2f}, {v.max():6.2f}]  "
              f"mean={v.mean():6.2f} V")


def diag_torque_comparison(torque_load: np.ndarray,
                           torque_urdf: np.ndarray,
                           nq: int):
    """DIAG 8: Compare Load Register vs URDF RNEA magnitudes."""
    print("\n" + "=" * 65)
    print("DIAG 8 — Torque magnitude comparison (RMS, N·m)")
    print("=" * 65)
    print(f"  {'joint':>5}  {'Load Reg':>10}  {'URDF':>10}  {'URDF/Load':>10}")
    print(f"  {'-----':>5}  {'--------':>10}  {'----':>10}  {'---------':>10}")

    for j in range(nq):
        rms_l = np.sqrt(np.mean(torque_load[:, j] ** 2))
        rms_u = np.sqrt(np.mean(torque_urdf[:, j] ** 2))
        ratio_ul = rms_u / rms_l if rms_l > 1e-9 else float('inf')
        print(f"  {j:>5d}  {rms_l:>10.4f}  {rms_u:>10.4f}  {ratio_ul:>10.2f}")


# ==================================================================
# Master function — run all diagnostics
# ==================================================================
def run_all(L: dict, M: dict, model, data,
            q: np.ndarray, qd: np.ndarray, qdd: np.ndarray,
            torque_load: np.ndarray,
            torque_urdf: np.ndarray):
    """
    Run every diagnostic in order.

    Parameters
    ----------
    L            : time-series arrays from data_loader
    M            : metadata dict from data_loader
    model, data  : Pinocchio model and data
    q, qd, qdd   : joint angles, velocities, accelerations
    torque_load  : measured torque from load register
    torque_urdf  : analytical torque from RNEA
    """
    nq  = model.nq
    dof = M["dof"]

    diag_pinocchio_joints(model, M["joint_map"])
    diag_urdf_masses(model)
    diag_gravity_torque(model, data, q)
    diag_joint_angles(q, nq)
    diag_timestamps(L["t"])
    diag_velocities(qd, qdd, nq)
    diag_raw_signals(L, dof)
    diag_torque_comparison(torque_load, torque_urdf, nq)

    print("\n" + "=" * 65)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 65)
