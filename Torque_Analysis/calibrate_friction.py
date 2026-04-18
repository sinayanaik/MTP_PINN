#!/usr/bin/env python3
"""
Calibrate friction parameters — final consolidated version.

Key improvements over v1/v2/v3:
  1. Bias-aware fitting: τ = b + c·tanh(q̇/ε) + v·q̇
     Bias 'b' absorbs mean gravity model error → cleaner c, v estimates
  2. Asymmetry analysis: split q̇>0 / q̇<0 → Coulomb robust to gravity error
  3. Per-trajectory consistency (bulk): fit per file → parameter stability
  4. Cross-method comparison: regime / sweep / nonlinear / asymmetry
  5. No velocity filtering (proven best with diverse trajectories)
  6. Analytical LS for sweep — no subsampling needed, instant on 470K samples

Code improvements:
  - Deterministic random seed for reproducible nonlinear subsampling
  - Logging instead of bare print for structured output
  - Uses config.FRICTION_EPS consistently as default

Always uses all files in raw_samples/ (bulk) — no CLI arguments needed.

Usage:
    cd ~/Desktop/MTP_PINN/Torque_Analysis
    python3 calibrate_friction.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob
import os
import time
import logging
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from Torque_Analysis import config as C
from Torque_Analysis.calibration_io import save_friction_params
from Torque_Analysis.data_loader import load_log
from Torque_Analysis.utils import ticks_to_radians
from Torque_Analysis.torque import (
    build_pinocchio_model,
    torque_from_load,
    torque_from_urdf,
)

logger = logging.getLogger(__name__)

# ==================================================================
# PHYSICAL BOUNDS — validated across v1/v2/v3 experiments
# ==================================================================
COULOMB_BOUNDS = (0.0, 0.50)       # N·m
VISCOUS_BOUNDS = (0.0, 0.30)       # N·m·s/rad
EPS_BOUNDS     = (0.02, 0.50)      # rad/s
EPS_SWEEP_RANGE = (0.02, 0.50)
EPS_SWEEP_POINTS = 100

# Nonlinear fit: subsample for speed (optimizer is slow)
NL_MAX_SAMPLES = 50000
NL_RANDOM_SEED = 42                # reproducible subsampling


# ==================================================================
# DATA LOADING
# ==================================================================
def load_friction_data(json_path, model):
    """Load one file → (qd, tau_f, q, run_id)."""
    L, M, N = load_log(json_path)
    nq = model.nq
    act_rad = ticks_to_radians(
        L["act_pos"], M["joint_map"], M["ticks_to_rad"], C.DOF
    )
    q = act_rad[:, :nq]
    pdata = model.createData()
    tau_load = torque_from_load(
        L["load"], L["voltage"], joint_map=M["joint_map"]
    )
    tau_urdf, qd, qdd = torque_from_urdf(model, pdata, q, L["t"])
    tau_fric_signal = tau_load[:, :nq] - tau_urdf
    return qd, tau_fric_signal, q, M["run_id"]


def load_all_data(model, bulk=False):
    """
    Load data. For per-trajectory analysis, also returns per-file arrays.

    Returns
    -------
    qd, tau_f, q : concatenated arrays (N_total, nq)
    per_file     : list of {"qd", "tau_f", "run_id"} dicts
    """
    per_file = []

    if bulk:
        raw_dir = os.path.join(C.PROJECT_ROOT, "raw_samples")
        json_files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
        print(f"\nLoading {len(json_files)} files...")

        qd_all, tau_f_all, q_all = [], [], []
        for jf in json_files:
            try:
                qd, tau_f, q, rid = load_friction_data(jf, model)
                qd_all.append(qd)
                tau_f_all.append(tau_f)
                q_all.append(q)
                per_file.append({"qd": qd, "tau_f": tau_f, "run_id": rid})
            except Exception as e:
                logger.warning("Skipping %s: %s", os.path.basename(jf)[:50], e)
                print(f"  ✗ {os.path.basename(jf)[:50]}: {e}")

        qd_cat = np.vstack(qd_all)
        tau_f_cat = np.vstack(tau_f_all)
        q_cat = np.vstack(q_all)
        print(f"  Loaded {len(per_file)}/{len(json_files)} files, "
              f"{qd_cat.shape[0]} total samples")
    else:
        print(f"\nLoading {os.path.basename(C.LOG_JSON)}...")
        qd_cat, tau_f_cat, q_cat, rid = load_friction_data(C.LOG_JSON, model)
        per_file.append({"qd": qd_cat, "tau_f": tau_f_cat, "run_id": rid})
        print(f"  Samples: {qd_cat.shape[0]}")

    return qd_cat, tau_f_cat, q_cat, per_file


# ==================================================================
# DIAGNOSTICS
# ==================================================================
def print_velocity_stats(qd, active):
    """Velocity percentiles — understand the data range."""
    print(f"\n" + "=" * 70)
    print("VELOCITY STATISTICS")
    print("=" * 70)
    print(f"\n  {'Joint':>5}  {'P50':>8}  {'P90':>8}  {'P95':>8}  "
          f"{'P99':>8}  {'Max':>8}  {'Samples':>10}")

    for j in range(active):
        absv = np.abs(qd[:, j])
        print(f"  {j:>5d}  {np.percentile(absv, 50):>8.3f}  "
              f"{np.percentile(absv, 90):>8.3f}  "
              f"{np.percentile(absv, 95):>8.3f}  "
              f"{np.percentile(absv, 99):>8.3f}  "
              f"{absv.max():>8.1f}  {len(absv):>10d}")


def correlation_diagnostic(qd, tau_f, q, active):
    """
    Check if friction signal correlates with position → gravity error leak.
    corr(τ_f, q̇) should be high (friction depends on velocity).
    corr(τ_f, q)  should be low (friction shouldn't depend on position).
    """
    print(f"\n" + "=" * 70)
    print("CORRELATION DIAGNOSTIC — Is gravity error leaking in?")
    print("=" * 70)
    print(f"\n  {'Joint':>5}  {'corr(τ_f, q̇)':>14}  {'corr(τ_f, q)':>14}  "
          f"{'Concern?':>10}")

    for j in range(active):
        r_vel = np.corrcoef(tau_f[:, j], qd[:, j])[0, 1]
        r_pos = np.corrcoef(tau_f[:, j], q[:, j])[0, 1]
        flag = "⚠ YES" if abs(r_pos) > 0.5 else "OK"
        print(f"  {j:>5d}  {r_vel:>+14.3f}  {r_pos:>+14.3f}  {flag:>10}")
        if abs(r_pos) > 0.5:
            logger.warning("Joint %d: high position correlation (%.3f) — gravity error leak", j, r_pos)

    print(f"\n  High |corr(τ_f, q)| → gravity model error in friction signal.")
    print(f"  Fix: use --bulk for trajectory diversity.")


def asymmetry_analysis(qd, tau_f, active, threshold=0.05):
    """
    Friction is antisymmetric: f(-q̇) = -f(q̇).
    Gravity error is NOT antisymmetric.

    Split data by velocity sign, fit τ = a + v·q̇ in each half:
      Positive half:  τ ≈ +c + v·q̇ + gravity_bias
      Negative half:  τ ≈ -c + v·q̇ + gravity_bias

    Decompose:
      Coulomb = (a_pos − a_neg) / 2    (antisymmetric → pure friction)
      Bias    = (a_pos + a_neg) / 2    (symmetric → gravity error)
      Viscous = (v_pos + v_neg) / 2

    This gives Coulomb estimates ROBUST to gravity model error.
    """
    print(f"\n" + "=" * 70)
    print("ASYMMETRY ANALYSIS — Separate friction from gravity error")
    print("=" * 70)
    print(f"\n  Split at |q̇| > {threshold} rad/s")
    print(f"\n  {'Joint':>5}  {'Coulomb':>10}  {'Viscous':>10}  "
          f"{'Bias':>10}  {'Bias/Coul':>10}  {'Interpretation':>20}")

    estimates = {}

    for j in range(active):
        pos = qd[:, j] > threshold
        neg = qd[:, j] < -threshold

        if pos.sum() < 50 or neg.sum() < 50:
            print(f"  {j:>5d}  — insufficient data (pos={pos.sum()}, "
                  f"neg={neg.sum()})")
            continue

        # Fit τ = a + v·q̇ in each half
        A_pos = np.column_stack([np.ones(pos.sum()), qd[pos, j]])
        x_pos = np.linalg.lstsq(A_pos, tau_f[pos, j], rcond=None)[0]

        A_neg = np.column_stack([np.ones(neg.sum()), qd[neg, j]])
        x_neg = np.linalg.lstsq(A_neg, tau_f[neg, j], rcond=None)[0]

        coulomb = (x_pos[0] - x_neg[0]) / 2
        bias    = (x_pos[0] + x_neg[0]) / 2
        viscous = (x_pos[1] + x_neg[1]) / 2

        coulomb = max(0.0, coulomb)
        viscous = np.clip(viscous, 0.0, VISCOUS_BOUNDS[1])

        ratio = abs(bias / coulomb) if coulomb > 0.01 else float('inf')
        interp = ("gravity-dominated" if ratio > 1.0
                   else "significant bias" if ratio > 0.3
                   else "clean friction")

        print(f"  {j:>5d}  {coulomb:>10.4f}  {viscous:>10.4f}  "
              f"{bias:>+10.4f}  {ratio:>10.2f}  {interp:>20}")

        estimates[j] = {
            "coulomb": coulomb,
            "viscous": viscous,
            "bias": bias,
        }

    return estimates


# ==================================================================
# FITTING — REGIME ANALYSIS  (direct physical estimates)
# ==================================================================
def regime_analysis(qd, tau_f, active):
    """
    Low-speed:  Coulomb ≈ median(|τ_f|) where |q̇| is small
    High-speed: Viscous ≈ slope of (τ_f − c·sign(q̇)) vs q̇
    """
    print(f"\n" + "=" * 70)
    print("REGIME ANALYSIS — Direct physical estimates")
    print("=" * 70)

    LOW  = 0.05
    HIGH = 0.10
    estimates = {}

    for j in range(active):
        low  = (np.abs(qd[:, j]) > 0.01) & (np.abs(qd[:, j]) < LOW)
        high = np.abs(qd[:, j]) > HIGH

        print(f"\n  Joint {j}: low={low.sum()}, high={high.sum()}")

        if low.sum() > 20:
            c_est = np.median(np.abs(tau_f[low, j]))
            print(f"    Coulomb: {c_est:.4f} N·m")
        else:
            c_est = 0.05
            print(f"    Coulomb: {c_est:.4f} (fallback)")

        if high.sum() > 50:
            resid = tau_f[high, j] - c_est * np.sign(qd[high, j])
            A = qd[high, j].reshape(-1, 1)
            v_est = float(np.linalg.lstsq(A, resid, rcond=None)[0][0])
            v_est = np.clip(v_est, 0.0, VISCOUS_BOUNDS[1])
            print(f"    Viscous: {v_est:.4f} N·m·s/rad")
        else:
            v_est = 0.01
            print(f"    Viscous: {v_est:.4f} (fallback)")

        estimates[j] = {"coulomb": c_est, "viscous": v_est}

    return estimates


# ==================================================================
# FITTING — BIAS-AWARE ε SWEEP  (analytical LS, instant)
# ==================================================================
def fit_bias_aware(qd_j, tau_f_j, eps):
    """
    Analytical LS:  τ_f = b + c·tanh(q̇/ε) + v·q̇
    b absorbs mean gravity error → cleaner c, v.
    """
    phi = np.column_stack([
        np.ones(len(qd_j)),
        np.tanh(qd_j / eps),
        qd_j,
    ])

    x, _, _, _ = np.linalg.lstsq(phi, tau_f_j, rcond=None)

    b = x[0]
    c = np.clip(x[1], *COULOMB_BOUNDS)
    v = np.clip(x[2], *VISCOUS_BOUNDS)

    pred = b + c * np.tanh(qd_j / eps) + v * qd_j
    rms_val = np.sqrt(np.mean((tau_f_j - pred) ** 2))
    return b, c, v, rms_val


def sweep_eps_bias(qd, tau_f, active):
    """
    Grid search: for each ε, fit (b, c, v) analytically per joint.
    Analytical LS on full dataset — no subsampling needed.
    """
    eps_range = np.linspace(*EPS_SWEEP_RANGE, EPS_SWEEP_POINTS)

    best_rms = np.inf
    best_eps = None
    best_params = None
    all_results = []

    for eps in eps_range:
        joint_rms_sq = 0.0
        params = []

        for j in range(active):
            b, c, v, rms_j = fit_bias_aware(qd[:, j], tau_f[:, j], eps)
            joint_rms_sq += rms_j ** 2
            params.append((b, c, v, rms_j))

        total_rms = np.sqrt(joint_rms_sq / active)
        all_results.append((eps, total_rms, params))

        if total_rms < best_rms:
            best_rms = total_rms
            best_eps = eps
            best_params = params

    return best_eps, best_params, best_rms, all_results


# ==================================================================
# FITTING — NONLINEAR PER-JOINT ε
# ==================================================================
def fit_nonlinear_joint(qd_j, tau_f_j,
                        c0=0.10, v0=0.02, eps0=None):
    """
    Full nonlinear: min ‖τ_f − c·tanh(q̇/ε) − v·q̇‖²
    Note: no bias term — this fits the model that goes into config.py.
    """
    if eps0 is None:
        eps0 = C.FRICTION_EPS

    # Deterministic subsample for optimizer speed
    rng = np.random.RandomState(NL_RANDOM_SEED)
    if len(qd_j) > NL_MAX_SAMPLES:
        idx = rng.choice(len(qd_j), NL_MAX_SAMPLES, replace=False)
        qd_s, tf_s = qd_j[idx], tau_f_j[idx]
    else:
        qd_s, tf_s = qd_j, tau_f_j

    def cost(p):
        return np.mean(
            (tf_s - p[0] * np.tanh(qd_s / p[2]) - p[1] * qd_s) ** 2
        )

    res = minimize(cost, x0=[c0, v0, eps0],
                   bounds=[COULOMB_BOUNDS, VISCOUS_BOUNDS, EPS_BOUNDS],
                   method='L-BFGS-B')

    c, v, eps = res.x
    pred = c * np.tanh(qd_j / eps) + v * qd_j
    rms_val = np.sqrt(np.mean((tau_f_j - pred) ** 2))
    return c, v, eps, rms_val


# ==================================================================
# FITTING — PER-TRAJECTORY CONSISTENCY  (bulk only)
# ==================================================================
def per_trajectory_consistency(per_file, active, eps):
    """
    Fit (b, c, v) per trajectory at fixed ε.
    Stable parameters → trustworthy.  High spread → suspicious.
    """
    if len(per_file) < 3:
        return None

    print(f"\n" + "=" * 70)
    print(f"PER-TRAJECTORY CONSISTENCY  "
          f"({len(per_file)} files, ε={eps:.4f})")
    print("=" * 70)

    all_params = {j: {"b": [], "c": [], "v": []} for j in range(active)}

    for pf in per_file:
        nq_file = pf["qd"].shape[1]
        for j in range(min(active, nq_file)):
            b, c, v, _ = fit_bias_aware(
                pf["qd"][:, j], pf["tau_f"][:, j], eps
            )
            all_params[j]["b"].append(b)
            all_params[j]["c"].append(c)
            all_params[j]["v"].append(v)

    print(f"\n  {'Joint':>5}  {'c mean':>8}  {'c std':>8}  {'c CV%':>7}  "
          f"{'v mean':>8}  {'v std':>8}  {'|b| mean':>8}  {'Stable?':>8}")

    stats = {}
    for j in range(active):
        cs = np.array(all_params[j]["c"])
        vs = np.array(all_params[j]["v"])
        bs = np.array(all_params[j]["b"])

        c_cv = 100 * cs.std() / max(cs.mean(), 1e-6)
        stable = "✓" if c_cv < 30 and vs.std() < 0.05 else "~"

        print(f"  {j:>5d}  {cs.mean():>8.4f}  {cs.std():>8.4f}  "
              f"{c_cv:>6.1f}%  {vs.mean():>8.4f}  {vs.std():>8.4f}  "
              f"{np.abs(bs).mean():>8.4f}  {stable:>8}")

        stats[j] = {
            "c_mean": cs.mean(), "c_std": cs.std(),
            "v_mean": vs.mean(), "v_std": vs.std(),
            "b_mean": bs.mean(), "b_std": bs.std(),
            "c_all": cs, "v_all": vs, "b_all": bs,
        }

    return stats


# ==================================================================
# SYNTHESIS — Cross-method comparison
# ==================================================================
def synthesize_recommendation(active, regime_est, asym_est,
                              sweep_params, sweep_eps,
                              nl_params, nl_eps,
                              traj_stats):
    """
    Compare all methods, pick robust values.

    Priority:
      1. Asymmetry Coulomb (most robust to gravity error)
      2. Sweep viscous (full dataset, analytical)
      3. Nonlinear ε (per-joint flexibility)
      4. Per-trajectory stats for confidence
    """
    print(f"\n" + "=" * 70)
    print("CROSS-METHOD COMPARISON")
    print("=" * 70)

    header = (f"  {'Joint':>5}  {'Regime c':>9}  {'Asym c':>9}  "
              f"{'Sweep c':>9}  {'NL c':>9}  │  "
              f"{'Regime v':>9}  {'Sweep v':>9}  {'NL v':>9}")
    print(f"\n{header}")
    print(f"  {'─' * 5}  {'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 9}  │  "
          f"{'─' * 9}  {'─' * 9}  {'─' * 9}")

    final_c = np.zeros(C.DOF)
    final_v = np.zeros(C.DOF)

    for j in range(active):
        rc = regime_est[j]["coulomb"] if j in regime_est else float('nan')
        rv = regime_est[j]["viscous"] if j in regime_est else float('nan')

        ac = asym_est[j]["coulomb"] if asym_est and j in asym_est else float('nan')

        sc = sweep_params[j]["c"] if sweep_params else float('nan')
        sv = sweep_params[j]["v"] if sweep_params else float('nan')

        nc = nl_params[j]["c"] if nl_params else float('nan')
        nv = nl_params[j]["v"] if nl_params else float('nan')

        print(f"  {j:>5d}  {rc:>9.4f}  {ac:>9.4f}  {sc:>9.4f}  "
              f"{nc:>9.4f}  │  {rv:>9.4f}  {sv:>9.4f}  {nv:>9.4f}")

        # --- Coulomb: prefer asymmetry (gravity-robust), fallback to sweep ---
        candidates_c = []
        if asym_est and j in asym_est and not np.isnan(ac) and ac > 0.01:
            candidates_c.append(("asym", ac, 3.0))     # highest weight
        if sweep_params:
            candidates_c.append(("sweep", sc, 2.0))
        if nl_params:
            candidates_c.append(("nl", nc, 1.5))
        if j in regime_est and not np.isnan(rc):
            candidates_c.append(("regime", rc, 1.0))

        # Weighted average
        if candidates_c:
            total_w = sum(w for _, _, w in candidates_c)
            final_c[j] = sum(val * w for _, val, w in candidates_c) / total_w
        else:
            final_c[j] = 0.10

        # --- Viscous: prefer sweep (full data, analytical) ---
        candidates_v = []
        if sweep_params and not np.isnan(sv):
            candidates_v.append(("sweep", sv, 3.0))
        if nl_params and not np.isnan(nv):
            candidates_v.append(("nl", nv, 2.0))
        if j in regime_est and not np.isnan(rv):
            candidates_v.append(("regime", rv, 1.0))
        if asym_est and j in asym_est:
            av = asym_est[j]["viscous"]
            candidates_v.append(("asym", av, 2.0))

        if candidates_v:
            total_w = sum(w for _, _, w in candidates_v)
            final_v[j] = sum(val * w for _, val, w in candidates_v) / total_w
        else:
            final_v[j] = 0.01

        # Clip to bounds
        final_c[j] = np.clip(final_c[j], *COULOMB_BOUNDS)
        final_v[j] = np.clip(final_v[j], *VISCOUS_BOUNDS)

    # --- ε: use nonlinear per-joint if available, else sweep ---
    if nl_eps is not None:
        # Use median of per-joint, excluding outlier values at bounds
        eps_vals = [nl_eps[j] for j in range(active)]
        interior = [e for e in eps_vals
                    if abs(e - EPS_BOUNDS[0]) > 0.005
                    and abs(e - EPS_BOUNDS[1]) > 0.005]
        if interior:
            final_eps = float(np.median(interior))
        else:
            final_eps = float(np.median(eps_vals))
    elif sweep_eps is not None:
        final_eps = sweep_eps
    else:
        final_eps = C.FRICTION_EPS  # use current config as fallback

    final_eps = np.clip(final_eps, *EPS_BOUNDS)

    # --- Confidence from per-trajectory consistency ---
    if traj_stats:
        print(f"\n  Per-trajectory stability (CV% of Coulomb):")
        for j in range(active):
            if j in traj_stats:
                cv = 100 * traj_stats[j]["c_std"] / max(
                    traj_stats[j]["c_mean"], 1e-6)
                conf = "HIGH" if cv < 20 else "MEDIUM" if cv < 40 else "LOW"
                print(f"    Joint {j}: CV={cv:.1f}% → confidence: {conf}")

    return final_c, final_v, final_eps


# ==================================================================
# PLOTS
# ==================================================================
def plot_eps_sweep(all_results, best_eps, title_suffix=""):
    eps_vals = [r[0] for r in all_results]
    rms_vals = [r[1] for r in all_results]

    fig, ax = plt.subplots(figsize=(10, 4), dpi=C.DPI)
    ax.plot(eps_vals, rms_vals, lw=1.5)
    ax.axvline(best_eps, color="red", ls="--",
               label=f"Best ε = {best_eps:.4f}")
    ax.axvline(C.FRICTION_EPS, color="orange", ls="--",
               label=f"Current ε = {C.FRICTION_EPS}")
    ax.set_xlabel("ε (rad/s)")
    ax.set_ylabel("Mean RMS Residual (N·m)")
    ax.set_title(f"ε Sweep — Bias-Aware Fit{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_friction_curves(qd, tau_f, active,
                         c_new, v_new, eps_new,
                         asym_est, method_name, bulk):
    """Scatter + new fit + old fit + asymmetry estimate."""
    fig, axes = plt.subplots(3, 2, figsize=(C.FIG_WIDTH, 14), dpi=C.DPI)

    rng = np.random.RandomState(NL_RANDOM_SEED)

    for j, ax in enumerate(axes.flat):
        if j >= active:
            ax.set_visible(False)
            continue

        # Subsample for plotting (deterministic)
        n_plot = min(len(qd), 30000)
        if len(qd) > n_plot:
            idx = rng.choice(len(qd), n_plot, replace=False)
        else:
            idx = np.arange(len(qd))

        ax.scatter(qd[idx, j], tau_f[idx, j],
                   s=0.3, alpha=0.05, color="gray", rasterized=True)

        # Curve range
        qd_lo = np.percentile(qd[:, j], 0.5)
        qd_hi = np.percentile(qd[:, j], 99.5)
        qd_range = np.linspace(qd_lo, qd_hi, 500)

        # New fit
        tau_new = (c_new[j] * np.tanh(qd_range / eps_new)
                   + v_new[j] * qd_range)
        ax.plot(qd_range, tau_new, lw=2.5, color="C2",
                label=f"New: c={c_new[j]:.3f} v={v_new[j]:.3f}")

        # Old config
        tau_old = (C.COULOMB_NM[j] * np.tanh(qd_range / C.FRICTION_EPS)
                   + C.VISCOUS_NM[j] * qd_range)
        ax.plot(qd_range, tau_old, lw=1.5, ls="--", color="C3",
                label=f"Old: c={C.COULOMB_NM[j]:.2f} v={C.VISCOUS_NM[j]:.2f}")

        # Asymmetry estimate
        if asym_est and j in asym_est:
            ae = asym_est[j]
            tau_asym = (ae["coulomb"] * np.sign(qd_range)
                        + ae["viscous"] * qd_range)
            ax.plot(qd_range, tau_asym, lw=1.0, ls=":", color="C4",
                    label=f"Asym: c={ae['coulomb']:.3f} b={ae['bias']:+.3f}")

        ax.axhline(0, color='k', lw=0.3)
        ax.axvline(0, color='k', lw=0.3)
        ax.set_title(f"Joint {j+1} (ε={eps_new:.3f})", fontsize=10)
        ax.set_xlabel("q̇ (rad/s)", fontsize=8)
        ax.set_ylabel("τ_load − τ_RNEA (N·m)", fontsize=8)
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

    src = f"bulk ({len(qd)} samples)" if bulk else "single file"
    fig.suptitle(
        f"Friction Calibration — {method_name}\n"
        f"Data: {src}  |  v ≤ {VISCOUS_BOUNDS[1]} N·m·s/rad",
        fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_bias_magnitude(sweep_params, active):
    """Bar chart: how much gravity error (bias) per joint."""
    if not sweep_params:
        return

    biases = [abs(sweep_params[j]["b"]) for j in range(active)]
    coulombs = [sweep_params[j]["c"] for j in range(active)]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=C.DPI)
    x = np.arange(active)
    w = 0.35
    ax.bar(x - w/2, coulombs, w, label="Coulomb (c)", color="C2", alpha=0.7)
    ax.bar(x + w/2, biases, w, label="|Bias| (|b|)", color="C3", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"J{j+1}" for j in x])
    ax.set_ylabel("N·m")
    ax.set_title("Coulomb vs Gravity Bias — Bias-Aware Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_per_joint_eps(eps_arr, active):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=C.DPI)
    joints = np.arange(active)
    ax.bar(joints, eps_arr[:active], color="C0", alpha=0.7)
    ax.axhline(C.FRICTION_EPS, color="orange", ls="--",
               label=f"Current ε = {C.FRICTION_EPS}")
    ax.axhline(EPS_BOUNDS[0], color="gray", ls=":",
               label=f"Min ε = {EPS_BOUNDS[0]}")
    ax.axhline(np.median(eps_arr[:active]), color="red", ls="--",
               label=f"Median = {np.median(eps_arr[:active]):.4f}")
    ax.set_xticks(joints)
    ax.set_xticklabels([f"J{j+1}" for j in joints])
    ax.set_ylabel("ε (rad/s)")
    ax.set_title("Per-Joint ε (Nonlinear Fit)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_per_trajectory_spread(traj_stats, active):
    """Box plot of per-trajectory Coulomb estimates."""
    if not traj_stats:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=C.DPI)

    # Coulomb spread
    data_c = [traj_stats[j]["c_all"] for j in range(active)
              if j in traj_stats]
    axes[0].boxplot(data_c, labels=[f"J{j+1}" for j in range(len(data_c))])
    axes[0].set_ylabel("Coulomb (N·m)")
    axes[0].set_title("Per-Trajectory Coulomb Spread")
    axes[0].grid(True, alpha=0.3, axis='y')

    # Viscous spread
    data_v = [traj_stats[j]["v_all"] for j in range(active)
              if j in traj_stats]
    axes[1].boxplot(data_v, labels=[f"J{j+1}" for j in range(len(data_v))])
    axes[1].set_ylabel("Viscous (N·m·s/rad)")
    axes[1].set_title("Per-Trajectory Viscous Spread")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def rms(x):
    return np.sqrt(np.mean(x ** 2))


# ==================================================================
# MAIN
# ==================================================================
def main():
    C.setup_logging()
    t_start = time.time()

    # ---- Build model ----
    print("Building Pinocchio model...")
    model, _, nq = build_pinocchio_model(
        C.XACRO_PATH, mass_scale=C.MASS_SCALE, extra_masses=C.EXTRA_MASSES,
    )
    active = min(nq, C.ACTIVE_JOINTS)
    print(f"  nq={nq}, active={active}")

    # ---- Load data ----
    qd, tau_f, q, per_file = load_all_data(model, bulk=True)

    # ==============================================================
    # DIAGNOSTICS  (run first — understand data before fitting)
    # ==============================================================
    print_velocity_stats(qd, active)
    correlation_diagnostic(qd, tau_f, q, active)

    # ==============================================================
    # METHOD A: REGIME ANALYSIS  (direct, no optimization)
    # ==============================================================
    regime_est = regime_analysis(qd, tau_f, active)

    # ==============================================================
    # METHOD B: ASYMMETRY ANALYSIS  (gravity-robust Coulomb)
    # ==============================================================
    asym_est = asymmetry_analysis(qd, tau_f, active)

    # ==============================================================
    # CURRENT BASELINE
    # ==============================================================
    print(f"\n  Current config baseline RMS:")
    for j in range(active):
        tau_old_j = (C.COULOMB_NM[j] * np.tanh(qd[:, j] / C.FRICTION_EPS)
                     + C.VISCOUS_NM[j] * qd[:, j])
        print(f"    Joint {j}: {rms(tau_f[:, j] - tau_old_j):.4f} N·m")

    # ==============================================================
    # METHOD C: BIAS-AWARE ε SWEEP  (analytical, instant)
    # ==============================================================
    sweep_params_dict = None
    best_sweep_eps = None

    if True:  # sweep (always run)
        print(f"\n" + "=" * 70)
        print("METHOD C: BIAS-AWARE ε SWEEP")
        print(f"  Model: τ = b + c·tanh(q̇/ε) + v·q̇")
        print(f"  Analytical LS — no subsampling needed")
        print("=" * 70)

        t0 = time.time()
        best_eps, best_params, best_rms, all_results = sweep_eps_bias(
            qd, tau_f, active
        )
        dt = time.time() - t0

        sweep_params_dict = {}
        print(f"\n  Best ε = {best_eps:.4f}  "
              f"(RMS = {best_rms:.6f} N·m, {dt:.1f}s)")

        print(f"\n  {'Joint':>5}  {'Bias(b)':>10}  {'Coulomb':>10}  "
              f"{'Viscous':>10}  {'RMS':>10}  {'Flags':>12}")
        print(f"  {'─'*5:>5}  {'─'*10:>10}  {'─'*10:>10}  "
              f"{'─'*10:>10}  {'─'*10:>10}  {'─'*12:>12}")

        for j in range(active):
            b, c, v, rms_j = best_params[j]
            sweep_params_dict[j] = {"b": b, "c": c, "v": v, "rms": rms_j}

            flags = []
            if abs(v - VISCOUS_BOUNDS[1]) < 1e-4:
                flags.append("v_cap")
            if abs(b) > 0.05:
                flags.append(f"bias")
            flag_str = ", ".join(flags) if flags else "—"

            print(f"  {j:>5d}  {b:>+10.4f}  {c:>10.4f}  "
                  f"{v:>10.4f}  {rms_j:>10.4f}  {flag_str:>12}")

        best_sweep_eps = best_eps
        plot_eps_sweep(all_results, best_eps)
        plot_bias_magnitude(sweep_params_dict, active)

    # ==============================================================
    # METHOD D: NONLINEAR PER-JOINT ε  (no bias — config.py model)
    # ==============================================================
    nl_params_dict = None
    nl_eps_arr = None

    if True:  # nonlinear (always run)
        print(f"\n" + "=" * 70)
        print("METHOD D: NONLINEAR PER-JOINT ε")
        print(f"  Model: τ = c·tanh(q̇/ε_j) + v·q̇  (no bias)")
        print(f"  This matches config.py / torque.py model")
        print("=" * 70)

        nl_params_dict = {}
        nl_eps_arr = np.zeros(C.DOF)

        print(f"\n  {'Joint':>5}  {'Coulomb':>10}  {'Viscous':>10}  "
              f"{'ε':>10}  {'RMS':>10}  {'Flags':>12}")
        print(f"  {'─'*5:>5}  {'─'*10:>10}  {'─'*10:>10}  "
              f"{'─'*10:>10}  {'─'*10:>10}  {'─'*12:>12}")

        for j in range(active):
            # Warm-start from sweep if available
            if sweep_params_dict:
                c0 = sweep_params_dict[j]["c"]
                v0 = sweep_params_dict[j]["v"]
                eps0 = best_sweep_eps
            else:
                c0, v0, eps0 = 0.10, 0.02, C.FRICTION_EPS

            c, v, eps_j, rms_j = fit_nonlinear_joint(
                qd[:, j], tau_f[:, j], c0, v0, eps0
            )
            nl_params_dict[j] = {"c": c, "v": v, "eps": eps_j, "rms": rms_j}
            nl_eps_arr[j] = eps_j

            flags = []
            if abs(v - VISCOUS_BOUNDS[1]) < 1e-4:
                flags.append("v_cap")
            if abs(eps_j - EPS_BOUNDS[0]) < 1e-4:
                flags.append("ε_floor")
            if abs(eps_j - EPS_BOUNDS[1]) < 1e-4:
                flags.append("ε_ceil")
            flag_str = ", ".join(flags) if flags else "—"

            print(f"  {j:>5d}  {c:>10.4f}  {v:>10.4f}  "
                  f"{eps_j:>10.4f}  {rms_j:>10.4f}  {flag_str:>12}")

        print(f"\n  Per-joint ε: "
              f"{np.round(nl_eps_arr[:active], 4).tolist()}")
        print(f"  Median ε:    "
              f"{np.median(nl_eps_arr[:active]):.4f}")

        plot_per_joint_eps(nl_eps_arr, active)

    # ==============================================================
    # PER-TRAJECTORY CONSISTENCY  (bulk only)
    # ==============================================================
    traj_stats = None
    if best_sweep_eps is not None:
        traj_stats = per_trajectory_consistency(
            per_file, active, best_sweep_eps
        )
        if traj_stats:
            plot_per_trajectory_spread(traj_stats, active)

    # ==============================================================
    # SYNTHESIS — Cross-method recommendation
    # ==============================================================
    final_c, final_v, final_eps = synthesize_recommendation(
        active, regime_est, asym_est,
        sweep_params_dict, best_sweep_eps,
        nl_params_dict, nl_eps_arr,
        traj_stats,
    )

    # ==============================================================
    # VALIDATION — RMS improvement
    # ==============================================================
    print(f"\n" + "=" * 70)
    print("VALIDATION — RMS IMPROVEMENT")
    print("=" * 70)

    print(f"\n  {'Joint':>5}  {'Old RMS':>10}  {'New RMS':>10}  "
          f"{'Δ%':>8}  {'c':>8}  {'v':>8}")

    for j in range(active):
        tau_old_j = (C.COULOMB_NM[j] * np.tanh(qd[:, j] / C.FRICTION_EPS)
                     + C.VISCOUS_NM[j] * qd[:, j])
        tau_new_j = (final_c[j] * np.tanh(qd[:, j] / final_eps)
                     + final_v[j] * qd[:, j])
        old_r = rms(tau_f[:, j] - tau_old_j)
        new_r = rms(tau_f[:, j] - tau_new_j)
        pct = (new_r - old_r) / old_r * 100 if old_r > 1e-9 else 0
        print(f"  {j:>5d}  {old_r:>10.4f}  {new_r:>10.4f}  "
              f"{pct:>+7.1f}%  {final_c[j]:>8.4f}  {final_v[j]:>8.4f}")

    # ==============================================================
    # FRICTION CURVES
    # ==============================================================
    plot_friction_curves(
        qd, tau_f, active,
        final_c, final_v, final_eps,
        asym_est, "Synthesized", True,
    )

    # ==============================================================
    # FINAL OUTPUT
    # ==============================================================
    total_time = time.time() - t_start

    print(f"\n" + "=" * 70)
    print("★ FINAL RECOMMENDATION FOR config.py")
    print("=" * 70)
    print(f"  COULOMB_NM   = np.array("
          f"{np.round(final_c, 4).tolist()})")
    print(f"  VISCOUS_NM   = np.array("
          f"{np.round(final_v, 4).tolist()})")
    print(f"  FRICTION_EPS = {final_eps:.4f}")

    # Warnings
    print(f"\n  Methods used: regime, asymmetry, "
          f"{'sweep, ' if sweep_params_dict else ''}"
          f"{'nonlinear, ' if nl_params_dict else ''}"
          f"{'per-traj consistency' if traj_stats else ''}")
    print(f"  Total time: {total_time:.1f}s")

    if asym_est:
        biased_joints = [j for j in range(active)
                         if j in asym_est
                         and abs(asym_est[j]["bias"]) > 0.05]
        if biased_joints:
            print(f"\n  ⚠ Joints {biased_joints} have significant "
                  f"gravity bias (|b| > 0.05 N·m).")

    # Report which joints hit the viscous cap — this is a physically meaningful signal.
    #
    # Hitting the cap means the optimizer could not find a valid minimum within bounds.
    # Two common root causes:
    #   (a) Wrong stall torque: τ_load is overestimated → τ_f = τ_load − τ_RNEA is too
    #       large → the viscous term expands to absorb the extra.
    #       Fix: run `calibrate_mass.py` and check the stall torque diagnostic.
    #   (b) Missing inertia: servo masses not in the URDF → M(q)·q̈ is underestimated
    #       → inertial error leaks into τ_f and looks like viscous drag.
    #       Fix: add EXTRA_MASSES in config.py for distal servo locations.
    # Raising VISCOUS_BOUNDS without fixing the root cause will absorb
    # more model error into the viscous term, not improve physical accuracy.
    v_cap_joints = [j for j in range(active)
                    if abs(final_v[j] - VISCOUS_BOUNDS[1]) < 1e-4]
    if v_cap_joints:
        joint_labels = [f"J{j+1}" for j in v_cap_joints]
        print(f"\n  ⚠ Joints at viscous cap ({VISCOUS_BOUNDS[1]} N·m·s/rad): "
              f"{joint_labels}")
        print(f"    Likely cause: wrong stall torque or unmodelled inertia.")
        print(f"    Recommended fix:")
        print(f"      1. Run: python3 calibrate_mass.py")
        print(f"         Check 'STALL TORQUE DIAGNOSTIC' in the output.")
        print(f"      2. Update STALL_TORQUE_PER_JOINT in config.py if instructed.")
        print(f"      3. Re-run this script with the corrected model.")

    # ==============================================================
    # SAVE TO calibration_params.json
    # ==============================================================
    # Collect per-joint RMS before/after for the history record
    rms_old_list = []
    rms_new_list = []
    for j in range(active):
        tau_old_j = (C.COULOMB_NM[j] * np.tanh(qd[:, j] / C.FRICTION_EPS)
                     + C.VISCOUS_NM[j] * qd[:, j])
        tau_new_j = (final_c[j] * np.tanh(qd[:, j] / final_eps)
                     + final_v[j] * qd[:, j])
        rms_old_list.append(float(rms(tau_f[:, j] - tau_old_j)))
        rms_new_list.append(float(rms(tau_f[:, j] - tau_new_j)))

    # Pad passive joint (J6) with zeros
    coulomb_full = list(final_c[:active]) + [0.0] * (C.DOF - active)
    viscous_full = list(final_v[:active]) + [0.0] * (C.DOF - active)

    raw_dir = os.path.join(C.PROJECT_ROOT, "raw_samples")
    n_files = len(glob.glob(os.path.join(raw_dir, "*.json")))
    source_info = f"bulk ({n_files} files, {qd.shape[0]} samples)"

    save_friction_params(
        coulomb_nm   = coulomb_full,
        viscous_nm   = viscous_full,
        friction_eps = final_eps,
        bulk         = True,
        n_samples    = int(qd.shape[0]),
        source_info  = source_info,
        rms_old      = rms_old_list,
        rms_new      = rms_new_list,
    )


if __name__ == "__main__":
    main()
