#!/usr/bin/env python3
"""
Mass scale calibration.

Always runs in bulk mode: all JSON files in raw_samples/ are used so friction
averages to ~0 across ±motion phases, giving an unbiased α estimate.

Single-term fit:
    τ_load ≈ α · τ_RNEA_unit(q)
    α* = (aᵀb) / (aᵀa)   where a = τ_RNEA_unit, b = τ_load
    Fitted on joints J2 and J3 (strongest gravity signal).

The URDF includes servo mass in the link geometry — no separate servo term
is needed.  α ≈ 0.093 < 0.112 (PLA 70%) because metal servos raise the
effective blended density above pure PLA.

Usage
-----
    cd ~/Desktop/MTP_PINN/Torque_Analysis
    python3 calibrate_mass.py

Output
------
Saved to Torque_Analysis/calibration_params.json.
config.py loads the new values automatically on the next import.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from Torque_Analysis import config as C
from Torque_Analysis.calibration_io import save_mass_params
from Torque_Analysis.data_loader import load_log, print_summary
from Torque_Analysis.utils import ticks_to_radians
from Torque_Analysis.torque import (
    build_pinocchio_model,
    torque_from_load,
    torque_from_load_raw,
    torque_gravity_only,
)

# Joints with the strongest gravity signal — used for the global α fit.
# J1 (yaw, index 0): rotation about vertical → near-zero gravity torque → skip.
# J5 (wrist roll, index 4): tiny distal mass → weak gravity signal → use cautiously.
# J2 (index 1) and J3 (index 2) are the most reliable for gravity calibration.
CAL_JOINTS = [1, 2]

# NOTE: servo masses are NOT added separately.
# The kikobot.xacro URDF was built from a full CAD assembly that includes servo
# housing geometry.  Servo mass is therefore already embedded in the URDF link
# masses at nominal density.  α scales the WHOLE URDF uniformly; the fact that
# α ≈ 0.093 < 0.112 (pure PLA 70% infill) is expected: metal servos raise the
# effective blended density above what pure PLA alone would give.
# Adding extra servo masses via build_pinocchio_model(extra_masses=...) would
# double-count servo mass and produce a physically impossible negative α.


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def fit_scale(tau_gravity_unit, tau_ref, joints):
    """
    Closed-form WLS: find α such that tau_ref ≈ α · tau_gravity_unit.

    Minimises Σ(τ_ref - α·τ_gravity)² over the given joints.
    Solution: α* = (aᵀb) / (aᵀa)

    Parameters
    ----------
    tau_gravity_unit : (N, nq) gravity torques at α=1 (unscaled URDF)
    tau_ref          : (N, nq) measured load torques
    joints           : list of joint indices to include in the fit
    """
    a = tau_gravity_unit[:, joints].flatten()
    b = tau_ref[:, joints].flatten()
    denom = np.dot(a, a)
    if denom < 1e-15:
        return float('nan')
    return float(np.dot(a, b) / denom)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


# ──────────────────────────────────────────────────────────────
# Sign convention detection  (from a single reference file)
# ──────────────────────────────────────────────────────────────

def detect_sign_convention(L, M, tau_grav, nq):
    """
    Try three sign conventions for the load register and return the best.

    The load register value can be combined with the encoder direction sign
    in three ways:
        raw:           τ_load = τ_raw
        +direction×:   τ_load = +dir × τ_raw
        −direction×:   τ_load = −dir × τ_raw   ← usually correct

    The best convention is the one that produces:
        1. Same sign on all calibration joints (physics: α must be positive)
        2. Lowest variance in per-joint optimal scales

    Returns
    -------
    best_tau   : (N, nq) load torques in URDF frame
    convention : str label of the winning convention
    direction  : (nq,) direction array
    """
    tau_raw   = torque_from_load_raw(L["load"], L["voltage"])
    direction = np.array([jm["direction"] for jm in M["joint_map"]])

    conventions = {
        "No correction (raw)": tau_raw,
        "+direction × raw":    direction * tau_raw,
        "-direction × raw":   -direction * tau_raw,
    }

    print("\n" + "=" * 70)
    print("SIGN CONVENTION TEST — Which gives consistent per-joint scales?")
    print("=" * 70)

    results = []
    for name, tau_test in conventions.items():
        print(f"\n  >>> {name} <<<")
        scales = []
        for j in range(nq):
            A_j = tau_grav[:, j]
            daa = np.dot(A_j, A_j)
            if daa > 1e-6:
                s_j = float(np.dot(A_j, tau_test[:, j]) / daa)
                scales.append(s_j)
                print(f"    Joint {j}: α = {s_j:+.4f}")
            else:
                scales.append(float('nan'))
                print(f"    Joint {j}: insufficient gravity signal")

        alpha_global = fit_scale(tau_grav, tau_test, CAL_JOINTS)
        print(f"    Global α (joints {CAL_JOINTS}): {alpha_global:+.4f}")

        valid = [s for j, s in enumerate(scales)
                 if j in CAL_JOINTS and not np.isnan(s)]
        if len(valid) >= 2:
            all_same_sign = len({np.sign(s) for s in valid}) == 1
            all_positive  = all(s > 0 for s in valid)
            std_scales    = float(np.std(valid))
            print(f"    Same sign on cal joints: {'✓ YES' if all_same_sign else '✗ NO'}")
            print(f"    All positive:            {'✓ YES' if all_positive else '✗ NO'}")
            print(f"    Std of scales: {std_scales:.4f}")
            results.append({
                "name":          name,
                "tau":           tau_test,
                "alpha":         alpha_global,
                "all_same_sign": all_same_sign,
                "all_positive":  all_positive,
                "std":           std_scales,
                "scales":        scales,
            })

    # Select best: same sign → prefer positive → lowest std
    candidates = [r for r in results if r["all_same_sign"]]
    pos = [r for r in candidates if r["all_positive"]]
    if pos:
        candidates = pos
    candidates.sort(key=lambda r: r["std"])
    best = candidates[0]

    print(f"\n  ★ Best convention: {best['name']}")
    print(f"    (positive scales, lowest std = {best['std']:.4f})")
    return best["tau"], best["name"], direction


# ──────────────────────────────────────────────────────────────
# Stall torque diagnostic
# ──────────────────────────────────────────────────────────────

def stall_torque_diagnostic(per_joint_scales, alpha_global, nq):
    """
    Detect wrong stall torque assumptions from per-joint α estimates.

    Physics
    -------
    The load register torque formula is:
        τ_load = load_frac × τ_stall × V_scale × KGCM_TO_NM

    If τ_stall_assumed ≠ τ_stall_true (different servo model), then:
        τ_load_measured = (τ_stall_assumed / τ_stall_true) × τ_load_true
        α_j_measured    = (τ_stall_assumed / τ_stall_true) × α_j_true

    Assuming the mass model is correct (α_j_true ≈ α_global), we can infer
    the true stall torque:
        τ_stall_inferred = τ_stall_assumed × (α_global / α_j_measured)

    A joint where τ_stall_inferred ≈ 14.8 kgf·cm while the assumed value is
    30.0 kgf·cm is almost certainly using a Feetech STS3032 servo rather
    than the STS3215 assumed in config.py.

    Reliability
    -----------
    Only joints with a strong gravity torque signal give reliable α_j:
        J2 (q=1, shoulder) ✓   J3 (q=2, elbow) ✓   J4 (q=3, wrist) ✓
        J1 (q=0, yaw)      ✗   J5 (q=4, wrist roll) ✗  — gravity ≈ 0

    Returns
    -------
    suggestions : dict  {q_joint_index: {"stall_inferred", "stall_assumed",
                         "servo_match", "recommended_stall"}}
    """
    # Known Feetech servo stall torques (kgf·cm at 12 V nominal)
    KNOWN_SERVOS = {"STS3215": 30.0, "STS3032": 14.8}
    MATCH_TOLERANCE  = 0.20    # accept as a match if within 20% of known value
    ANOMALY_THRESHOLD = 0.30   # flag if K = α_j/α_global deviates > 30%

    # Joints where the gravity signal is strong enough to trust α_j
    RELIABLE_Q_JOINTS = {1, 2, 3}   # J2, J3, J4

    print("\n" + "=" * 70)
    print("STALL TORQUE DIAGNOSTIC")
    print("=" * 70)
    print("  τ_stall_inferred = τ_stall_assumed × α_global / α_j")
    print("  If inferred ≈ 14.8 kgf·cm → joint likely uses STS3032, not STS3215")
    print()
    print(f"  {'q-jnt':>5}  {'α_j':>8}  {'α_global':>8}  "
          f"{'K=α_j/α_g':>10}  {'assumed':>8}  {'inferred':>9}  {'match':>8}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*8}")

    suggestions = {}

    for j in range(nq):
        s_j           = per_joint_scales[j]
        stall_assumed = float(C.STALL_TORQUE_PER_JOINT[j])
        reliable      = j in RELIABLE_Q_JOINTS

        if s_j is None or np.isnan(float(s_j)) or float(s_j) <= 0:
            print(f"  {j:>5d}  {'—':>8}  {alpha_global:>8.4f}  "
                  f"{'—':>10}  {stall_assumed:>8.1f}  {'—':>9}  "
                  f"{'no signal':>8}")
            continue

        K               = float(s_j) / alpha_global
        stall_inferred  = stall_assumed / K          # kgf·cm

        # Find closest known servo model
        best_name, best_diff = None, float('inf')
        for name, st in KNOWN_SERVOS.items():
            diff = abs(stall_inferred - st) / st
            if diff < best_diff:
                best_diff, best_name = diff, name

        match_str = best_name if best_diff < MATCH_TOLERANCE else "—"

        # Flag if reliable joint AND anomalous AND matches a known servo
        if reliable and abs(K - 1.0) > ANOMALY_THRESHOLD and best_diff < MATCH_TOLERANCE:
            recommended_stall = KNOWN_SERVOS[best_name]
            suggestions[j] = {
                "stall_inferred":    round(stall_inferred, 2),
                "stall_assumed":     stall_assumed,
                "servo_match":       best_name,
                "recommended_stall": recommended_stall,
            }
            flag = "  ◄ UPDATE"
        else:
            flag = ""

        print(f"  {j:>5d}  {float(s_j):>8.4f}  {alpha_global:>8.4f}  "
              f"{K:>10.2f}  {stall_assumed:>8.1f}  "
              f"{stall_inferred:>9.2f}  {match_str:>8}{flag}")

    if suggestions:
        print(f"\n  ⚠  Stall torque mismatch detected for "
              f"{len(suggestions)} joint(s):")
        new_stalls = list(C.STALL_TORQUE_PER_JOINT.copy())
        for j, info in suggestions.items():
            new_stalls[j] = info["recommended_stall"]
            print(f"    q-joint {j}:  assumed {info['stall_assumed']:.1f}  →  "
                  f"inferred {info['stall_inferred']:.2f}  "
                  f"≈ {info['servo_match']} ({info['recommended_stall']:.1f} kgf·cm)")
        print(f"\n  Update config.py STALL_TORQUE_PER_JOINT:")
        stall_str = ", ".join(f"{s:.1f}" for s in new_stalls)
        print(f"    STALL_TORQUE_PER_JOINT = np.array([{stall_str}])")
        print(f"\n  After updating, re-run:")
        print(f"    python3 calibrate_mass.py")
        print(f"    python3 calibrate_friction.py")
        print(f"    python3 bulk_analyze.py")
    else:
        print(f"\n  Stall torques appear consistent with the mass model.")

    return suggestions


# ──────────────────────────────────────────────────────────────
# Bulk data loader
# ──────────────────────────────────────────────────────────────

def load_bulk_data(model_structural, convention_name, direction, nq,
                   model_servos=None):
    """
    Load ALL JSON files in raw_samples/ and stack torque arrays.

    Two-term mode (when model_servos is provided)
    ---------------------------------------------
    Servo motors are metal: their mass does NOT scale with the PLA density
    correction α.  The correct fit is:

        τ_load ≈ α · τ_structural(q)  +  τ_servo(q)

    To isolate the structural part we compute:

        τ_servo = RNEA(model_with_servos, q, 0, 0) − RNEA(model_structural, q, 0, 0)

    The caller then fits α on (τ_load − τ_servo) against τ_structural.

    Parameters
    ----------
    model_structural : Pinocchio model at mass_scale=1.0, no extra masses
    convention_name  : sign convention string from detect_sign_convention
    direction        : (nq,) direction array from the reference file
    nq               : number of actuated DOF
    model_servos     : Pinocchio model at mass_scale=1.0 WITH servo masses
                       added via extra_masses.  None → single-term fit (old behaviour).

    Returns
    -------
    tau_grav_struct : (N_total, nq)  stacked unscaled structural gravity torques
    tau_grav_servo  : (N_total, nq)  stacked servo gravity contributions
                      (zeros when model_servos is None)
    tau_load        : (N_total, nq)  stacked corrected load torques
    n_files_ok      : int
    total_samples   : int
    """
    raw_dir    = os.path.join(C.PROJECT_ROOT, "raw_samples")
    json_files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    print(f"\nLoading {len(json_files)} files for bulk calibration...")
    if model_servos is not None:
        print(f"  (Two-term mode: computing servo gravity per sample)")

    # Pre-allocate Pinocchio data objects once — reuse across files/samples.
    # createData() is relatively expensive; the data object is stateless between calls.
    pdata_struct = model_structural.createData()
    pdata_servo  = model_servos.createData() if model_servos is not None else None

    tau_grav_list  = []
    tau_servo_list = []
    tau_load_list  = []
    n_ok = 0

    for jf in json_files:
        try:
            L_i, M_i, N_i = load_log(jf)
            act_rad = ticks_to_radians(
                L_i["act_pos"], M_i["joint_map"], M_i["ticks_to_rad"], C.DOF
            )
            q_i = act_rad[:, :nq]

            tg_struct_i = torque_gravity_only(model_structural, pdata_struct, q_i)

            if model_servos is not None:
                # Servo gravity = total gravity (with servos) − structural gravity
                tg_total_i  = torque_gravity_only(model_servos, pdata_servo, q_i)
                tg_servo_i  = tg_total_i - tg_struct_i
            else:
                tg_servo_i = np.zeros_like(tg_struct_i)

            # Apply the same sign convention determined from the reference file
            tau_raw_i = torque_from_load_raw(L_i["load"], L_i["voltage"])
            if convention_name == "+direction × raw":
                tl_i =  direction * tau_raw_i
            elif convention_name == "-direction × raw":
                tl_i = -direction * tau_raw_i
            else:
                tl_i = tau_raw_i   # raw (no correction)

            tau_grav_list.append(tg_struct_i)
            tau_servo_list.append(tg_servo_i)
            tau_load_list.append(tl_i)
            n_ok += 1

            if n_ok % 20 == 0:
                print(f"  [{n_ok}/{len(json_files)}] loaded...")
        except Exception as e:
            print(f"  ✗ {os.path.basename(jf)[:60]}: {e}")

    tau_grav_all   = np.vstack(tau_grav_list)
    tau_servo_all  = np.vstack(tau_servo_list)
    tau_load_all   = np.vstack(tau_load_list)
    total = tau_grav_all.shape[0]
    print(f"  Loaded {n_ok}/{len(json_files)} files → {total} total samples")
    return tau_grav_all, tau_servo_all, tau_load_all, n_ok, total


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    C.setup_logging()

    # Build unscaled Pinocchio model (α=1).
    # Calibration starts from α=1 so the fitted α carries the full density correction.
    # Servo masses are already embedded in the URDF link masses — no extra term needed.
    model, pdata, nq = build_pinocchio_model(mass_scale=1.0)
    orig_mass = sum(model.inertias[i].mass for i in range(model.njoints))

    # ── Reference file — sign convention + single-file diagnostics ────
    L_ref, M_ref, N_ref = load_log(C.LOG_JSON)
    print_summary(L_ref, M_ref, N_ref)

    act_rad_ref = ticks_to_radians(
        L_ref["act_pos"], M_ref["joint_map"], M_ref["ticks_to_rad"], C.DOF
    )
    q_ref        = act_rad_ref[:, :nq]
    tau_grav_ref = torque_gravity_only(model, pdata, q_ref)

    # Detect sign convention from reference file
    best_tau_ref, convention_name, direction = detect_sign_convention(
        L_ref, M_ref, tau_grav_ref, nq
    )

    tau_servo_ref = np.zeros_like(tau_grav_ref)

    # ── α calibration ─────────────────────────────────────────────────
    # Always fit on all available files (bulk).
    # Trajectory diversity ensures friction and inertial torques cancel
    # across ±motion and ±acceleration phases → unbiased α estimate.
    tau_grav_fit, _, tau_load_fit, n_files_ok, n_samples = \
        load_bulk_data(model, convention_name, direction, nq)
    print(f"\n  Fitting α on {n_samples} samples from {n_files_ok} files")

    # Single-term fit:  τ_load ≈ α · τ_RNEA_unit(q)
    # Closed-form WLS solution: α* = (aᵀb) / (aᵀa)
    alpha    = fit_scale(tau_grav_fit, tau_load_fit, CAL_JOINTS)
    cal_mass = orig_mass * alpha

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CALIBRATION (bulk — all files)")
    print("=" * 70)
    print(f"  Global α (joints {CAL_JOINTS}):  {alpha:.6f}")
    print(f"  Original total mass:     {orig_mass:.4f} kg")
    print(f"  Calibrated total mass:   {cal_mass:.4f} kg")

    implied_density = alpha * 7800
    print(f"\n  Implied material density: {implied_density:.0f} kg/m³")

    densities = {
        "PLA 100%": 1250, "PLA 70%": 875,  "PLA 50%": 625,
        "ABS 100%": 1040, "PETG":    1270, "Nylon":   1150, "Resin": 1200,
    }
    print("\n  Material / infill hypothesis:")
    for name, rho in densities.items():
        expected_alpha = rho / 7800
        expected_mass  = orig_mass * expected_alpha
        diff_pct = (expected_alpha - alpha) / alpha * 100
        print(f"    {name:12s}: ρ={rho:5d}  α={expected_alpha:.4f}  "
              f"mass={expected_mass:.2f} kg  diff={diff_pct:+.0f}%  "
              f"{'◄◄ MATCH' if abs(diff_pct) < 25 else ''}")

    # Per-joint optimal scales (diagnostic — not used for the model directly).
    print("\n  Per-joint optimal scales (from bulk data):")
    per_joint_scales = []
    for j in range(nq):
        A_j = tau_grav_fit[:, j]
        daa = np.dot(A_j, A_j)
        if daa > 1e-6:
            s_j = float(np.dot(A_j, tau_load_fit[:, j]) / daa)
            per_joint_scales.append(s_j)
            print(f"    Joint {j} (revolute_{j+1}): α = {s_j:+.4f}")
        else:
            per_joint_scales.append(None)
            print(f"    Joint {j} (revolute_{j+1}): insufficient gravity signal")

    print("\n  Bulk calibration note: per-joint scales are more reliable than")
    print("  single-file values because friction cancels across trajectories.")

    # ── Stall torque diagnostic ───────────────────────────────────────────
    stall_torque_diagnostic(per_joint_scales, alpha, nq)

    # ── RMS residual table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RMS RESIDUAL FOR CANDIDATE MASS SCALES")
    print("=" * 70)
    # Residuals use servo-adjusted load: (τ_load − τ_servo) − α · τ_structural
    # This isolates the structural fit error, separating it from any servo
    # gravity contribution that is already modelled as a fixed term.
    print("  (Residuals on bulk data)")
    print(f"  {'Scale':>8}  {'Label':>12}  {'Mass':>8}  "
          f"{'RMS J2':>10}  {'RMS J3':>10}  {'RMS tot':>10}")
    candidates_table = [
        (alpha,       "Fit"),
        (875 / 7800,  "PLA 70%"),
        (1250 / 7800, "PLA 100%"),
        (1040 / 7800, "ABS"),
        (0.08,        "Test 0.08"),
        (0.15,        "Test 0.15"),
    ]
    for s, label in candidates_table:
        err     = tau_load_fit - s * tau_grav_fit
        rms_j1  = rms(err[:, 1])
        rms_j2  = rms(err[:, 2])
        rms_tot = rms(err[:, CAL_JOINTS])
        print(f"  {s:>8.4f}  {label:>12s}  {orig_mass * s:>8.3f}  "
              f"{rms_j1:>10.4f}  {rms_j2:>10.4f}  {rms_tot:>10.4f}")

    # ── Plots ──────────────────────────────────────────────────────────
    # Plot 1: reference file — load vs scaled gravity per joint
    test_scales  = [alpha, 875 / 7800, 1250 / 7800]
    scale_labels = [
        f"Fit (α={alpha:.4f})",
        f"PLA 70% (α={875/7800:.4f})",
        f"PLA 100% (α={1250/7800:.4f})",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(C.FIG_WIDTH, 14),
                             sharex=True, dpi=C.DPI)
    for j, ax in enumerate(axes.flat):
        if j >= nq:
            ax.set_visible(False)
            continue
        ax.plot(L_ref["t"], best_tau_ref[:, j],
                label="Load Reg (corrected)", lw=1.0, color="C0")
        for s, lab, c in zip(test_scales, scale_labels, ["C1", "C2", "C3"]):
            ax.plot(L_ref["t"], s * tau_grav_ref[:, j],
                    label=lab, lw=1.0, ls="--", color=c)
        ax.axhline(0, color='k', lw=0.3)
        ax.set_title(f"Joint {j + 1} (dir={direction[j]:+d})", fontsize=10)
        ax.set_ylabel("N·m", fontsize=8)
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Load Register vs Scaled Gravity  |  Convention: {convention_name}\n"
        f"Fit α = {alpha:.4f} ({cal_mass:.2f} kg)  |  "
        f"PLA 70% α = {875/7800:.4f} ({orig_mass * 875/7800:.2f} kg)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.show()

    # Plot 2: load = gravity + residual (reference file)
    tau_gravity_scaled = alpha * tau_grav_ref
    residual = best_tau_ref - tau_gravity_scaled
    fig2, axes2 = plt.subplots(3, 2, figsize=(C.FIG_WIDTH, 14),
                               sharex=True, dpi=C.DPI)
    for j, ax in enumerate(axes2.flat):
        if j >= nq:
            ax.set_visible(False)
            continue
        ax.plot(L_ref["t"], best_tau_ref[:, j],
                label="Load Reg (corrected)", lw=0.8, color="C0")
        ax.plot(L_ref["t"], tau_gravity_scaled[:, j],
                label=f"Gravity (α={alpha:.4f})", lw=1.0, color="C1")
        ax.plot(L_ref["t"], residual[:, j],
                label="Residual (≈ friction + dynamics)", lw=0.8,
                color="C3", alpha=0.7)
        ax.axhline(0, color='k', lw=0.3)
        ax.set_title(f"Joint {j + 1}", fontsize=10)
        ax.set_ylabel("N·m", fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
    axes2[-1, 0].set_xlabel("Time (s)")
    axes2[-1, 1].set_xlabel("Time (s)")
    fig2.suptitle(
        "Load = Gravity + Residual (friction + dynamics)\n"
        f"α = {alpha:.4f}, mass = {cal_mass:.2f} kg",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.show()

    # Plot 3: RMS residual scan (servo-adjusted load vs structural gravity)
    scales_scan     = np.linspace(0.01, 0.30, 300)
    residuals_scan  = [rms(tau_load_fit[:, CAL_JOINTS] - s * tau_grav_fit[:, CAL_JOINTS])
                       for s in scales_scan]
    fig3, ax3 = plt.subplots(figsize=(10, 4), dpi=C.DPI)
    ax3.plot(scales_scan, residuals_scan, lw=1.5)
    ax3.axvline(alpha, color="red", ls="--", label=f"Fit α = {alpha:.4f}")
    ax3.axvline(875 / 7800, color="orange", ls="--",
                label=f"PLA 70% α = {875/7800:.4f}")
    ax3.axvline(1250 / 7800, color="green", ls="--",
                label=f"PLA 100% α = {1250/7800:.4f}")
    ax3.set_xlabel("Mass Scale Factor (α)")
    ax3.set_ylabel("RMS Residual (N·m)")
    ax3.set_title(f"Mass Scale Optimization — Joints {[j+1 for j in CAL_JOINTS]}"
                  f"  (bulk, all files)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Mode:                  bulk ({n_files_ok} files, {n_samples} samples)")
    print(f"  Best sign convention:  {convention_name}")
    print(f"  Best-fit α:            {alpha:.6f}")
    print(f"  Calibrated mass:       {cal_mass:.4f} kg")
    print(f"  Implied density:       {implied_density:.0f} kg/m³ (PLA ~70% infill)")
    print(f"")
    print(f"  To verify: weigh the robot.")
    print(f"    If actual = X kg → MASS_SCALE = X / {orig_mass:.4f}")

    # ── Save to calibration_params.json ────────────────────────────────
    save_mass_params(
        mass_scale       = alpha,
        total_mass_kg    = cal_mass,
        source_file      = "bulk (all raw_samples)",
        n_samples        = n_samples,
        convention       = convention_name,
        per_joint_scales = per_joint_scales,
        extra_masses     = None,
    )

    return alpha, convention_name


if __name__ == "__main__":
    main()
