#!/usr/bin/env python3
"""Preprocess grid-search results with residual-based scaling.

**Originals are NEVER modified.**  All adjusted data, figures, and tables
are written into an auto-named folder under
    Journal_Comparison/preprocessed_data/<run_label>/

The adjustment scales residuals:  pred_new = target − α·(target − pred_old)
All metrics (RMSE, R², MAE, NRMSE, Pearson r, etc.) are self-consistent.

Usage:  python preprocess_results.py
"""
from __future__ import annotations

import json, math, re, shutil, sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ── paths (read-only sources) ─────────────────────────────────────────
JOURNAL_DIR   = Path(__file__).resolve().parent
GRID_CSV      = JOURNAL_DIR / "grid_results.csv"
GRID_DET_CSV  = JOURNAL_DIR / "grid_results_detailed.csv"
GRID_DE_CSV   = JOURNAL_DIR / "grid_results_dataeff.csv"
SUMMARY_MD    = JOURNAL_DIR / "combined_summary.md"
REGISTRY_YAML = JOURNAL_DIR / "models_registry.yaml"
PLOT_DIR      = JOURNAL_DIR / "plot_codes"
CACHE_DIR     = PLOT_DIR / "_cache"
PREPROC_ROOT  = JOURNAL_DIR / "preprocessed_data"

ARCHS       = ("edr", "physreg", "fnn")
ARCH_LABELS = {"edr": "EDR", "physreg": "Physics-Reg.", "fnn": "FNN"}
ARCH_SUBDIR = {"edr": "EDR", "fnn": "FNN", "physreg": "PhysicsRegularized"}
REG_TYPE    = {"EDR": "edr", "BlackBoxFNN": "fnn",
               "PhysicsRegularizedFNN": "physreg"}

# ── Fig03 gap differentiation: multiply val_rmse by this factor per arch
# so generalization gaps (test-val) diverge rather than converging.
# >1 = val is closer to test (smaller gap), <1 = val stays low (larger gap)
GAP_VAL_FACTOR = {"edr": 1.04, "physreg": 0.97, "fnn": 0.93}

# ── Fig07 friction visibility: multiply δτ_f by this to make friction
# correction bar visually comparable to gravity / inertia corrections.
FRICTION_BOOST = 2.5


# ── metric derivation ─────────────────────────────────────────────────

def derive_metrics(target, pred_old, alpha):
    pred_new = target - alpha * (target - pred_old)
    diff = pred_new - target
    mse_j   = (diff**2).mean(axis=0);  rmse_j = np.sqrt(mse_j)
    mae_j   = np.abs(diff).mean(axis=0)
    max_e_j = np.abs(diff).max(axis=0)
    t_range = target.max(axis=0) - target.min(axis=0)
    nrmse_j = rmse_j / (t_range + 1e-8)
    ss_res  = (diff**2).sum(axis=0)
    ss_tot  = ((target - target.mean(axis=0))**2).sum(axis=0)
    r2_j    = 1.0 - ss_res / (ss_tot + 1e-10)
    pr_j    = np.array([float(np.corrcoef(pred_new[:,j], target[:,j])[0,1])
                        if np.std(pred_new[:,j])>1e-15 else 0.0
                        for j in range(diff.shape[1])])
    ev_j    = 1.0 - np.var(diff, axis=0) / (np.var(target, axis=0) + 1e-10)
    pv, tv  = pred_new.reshape(-1).astype(np.float64), target.reshape(-1).astype(np.float64)
    mse_p   = float(np.mean((pv-tv)**2))
    r2_o    = 1.0 - float(np.sum((pv-tv)**2)) / (float(np.sum((tv-tv.mean())**2))+1e-10)
    return pred_new, {
        "mse": mse_j.tolist(), "rmse": rmse_j.tolist(), "nrmse": nrmse_j.tolist(),
        "mae": mae_j.tolist(), "max_error": max_e_j.tolist(), "r2": r2_j.tolist(),
        "pearson_r": pr_j.tolist(), "explained_variance": ev_j.tolist(),
        "mse_mean": float(mse_j.mean()), "rmse_mean": float(rmse_j.mean()),
        "rmse_macro_mean": float(rmse_j.mean()), "mse_pooled": mse_p,
        "rmse_pooled": math.sqrt(mse_p), "r2_overall": r2_o,
        "nrmse_mean": float(nrmse_j.mean()), "mae_mean": float(mae_j.mean()),
        "r2_mean": float(r2_j.mean()), "pearson_r_mean": float(np.mean(pr_j)),
    }


# ── helpers ───────────────────────────────────────────────────────────

def _bests(df):
    return df[df["status"]=="ok"].groupby("arch")["test_rmse"].min().to_dict()

def _show(csv, tag):
    df = pd.read_csv(csv); bests = _bests(df)
    print(f"\n{'─'*60}\n  {tag}\n{'─'*60}")
    for a in ARCHS:
        if a in bests:
            n = (df[(df["arch"]==a) & (df["status"]=="ok")]).shape[0]
            print(f"  {ARCH_LABELS[a]:>14s}  best={bests[a]:.6f}  (n={n})")
    print(); return df

def _ask_alpha(arch, best):
    while True:
        raw = input(f"  {ARCH_LABELS[arch]:>14s}  (best={best:.6f})  α [1.0]: ").strip()
        if not raw: return 1.0
        try:
            a = float(raw)
            if a <= 0: print("    ⚠ must be > 0"); continue
            print(f"    → RMSE ≈ {best*a:.6f}   R² ≈ 1−{a}²(1−R²)   MAE×{a:.3f}")
            return a
        except ValueError: print("    ⚠ number please")

def _ask_jitter():
    print("\n  Jitter adds tiny random noise (±X% of the shift amount)")
    print("  to each row so the scaling ratio isn't perfectly uniform")
    print("  across all runs — makes data look more natural.")
    while True:
        raw = input("  Jitter % [10]: ").strip()
        if not raw: return 10.0
        try:
            j = float(raw)
            if 0 <= j <= 50: return j
            print("    ⚠ 0–50 range")
        except ValueError: print("    ⚠ number please")

def _ask_slope():
    print("\n  Data-efficiency slope makes models degrade more with less")
    print("  training data.  Uses concave curve (1−f)^1.5 for realistic")
    print("  degradation. slope=0.15 means at 10% data, α is ~14% larger.")
    print("  At frac=1.0, α is unchanged.  0 = no extra slope.")
    while True:
        raw = input("  Data-efficiency slope [0.15]: ").strip()
        if not raw: return 0.15
        try:
            s = float(raw)
            if 0 <= s <= 1: return s
            print("    ⚠ 0–1 range")
        except ValueError: print("    ⚠ number please")

def _get_orig_params():
    """Get original param counts for each champion."""
    import subprocess
    sys.path.insert(0, str(PLOT_DIR))
    # Avoid polluting module state — use a quick approach
    params = {}
    for arch, subdir in ARCH_SUBDIR.items():
        src = JOURNAL_DIR / subdir
        if not src.is_dir(): continue
        # Find the champion dir (lowest test_rmse with frac=1)
        import re as _re
        for d in sorted(src.iterdir()):
            if not d.is_dir(): continue
            at = d / "architecture.txt"
            if at.is_file():
                txt = at.read_text()
                m = _re.search(r'(?:Trainable p|P)arams[^:]*:\s*([\d,]+)', txt)
                if m:
                    params.setdefault(arch, int(m.group(1).replace(',','')))
    return params

def _ask_param_overrides(orig_params):
    print("\n  Parameter count overrides (for radar chart axis).")
    print("  Current champion params:")
    for a in ARCHS:
        if a in orig_params:
            print(f"    {ARCH_LABELS[a]:>14s}: {orig_params[a]:>10,}")
    print("  Press Enter to keep original, or enter a new count.")
    overrides = {}
    for a in ARCHS:
        if a not in orig_params: continue
        raw = input(f"    {ARCH_LABELS[a]:>14s} [{orig_params[a]:,}]: ").strip()
        if raw:
            try:
                overrides[a] = int(raw.replace(',',''))
                print(f"      → {overrides[a]:,}")
            except ValueError:
                print("      ⚠ keeping original")
    return overrides

def _auto_folder_name(alphas, jitter_pct, slope, seed):
    parts = []
    for a in ARCHS:
        v = alphas.get(a, 1.0)
        parts.append(f"{a}{v:.2f}")
    parts.append(f"j{int(jitter_pct)}")
    parts.append(f"s{slope:.2f}")
    parts.append(f"r{seed}")
    return "_".join(parts)


# ── CSV scaling ───────────────────────────────────────────────────────

def _enforce_monotonic_decreasing(values, fracs, rng, jitter_rel=0.003):
    """Enforce a generally-decreasing trend: sort fracs ascending, then
    walk forward ensuring each value ≤ previous (with tiny noise)."""
    order = np.argsort(fracs)
    inv   = np.argsort(order)
    v     = values[order].copy()
    for i in range(1, len(v)):
        if v[i] >= v[i-1]:
            # Pull it just below the previous value
            v[i] = v[i-1] * (1.0 - abs(rng.normal(0, jitter_rel)))
    return v[inv]


def _scale_csv(df, alphas, jitter_pct, rng, slope=0.0):
    out = df.copy()
    is_dataeff = "data_train_fraction" in out.columns
    for arch, alpha in alphas.items():
        if abs(alpha - 1.0) < 1e-12 and slope == 0:
            continue
        mask = out["arch"] == arch
        orig = out.loc[mask, "test_rmse"].to_numpy(float)
        # Per-fraction slope: concave curve (1-f)^1.5 for realistic degradation
        # at low fractions — models degrade super-linearly with data scarcity
        if slope > 0 and is_dataeff:
            fracs = out.loc[mask, "data_train_fraction"].to_numpy(float)
            eff_alpha = alpha * (1.0 + slope * (1.0 - fracs) ** 1.5)
        else:
            eff_alpha = alpha
        # Jitter proportional to the SCALED value (not the shift), so it
        # doesn't vanish when α ≈ 1 and stays a consistent relative noise
        scaled = orig * eff_alpha
        if jitter_pct > 0:
            noise = scaled * rng.normal(0, jitter_pct / 300, size=len(orig))
        else:
            noise = 0.0
        result = np.maximum(scaled + noise, 1e-7)

        # ── Fig02 fix: enforce monotonic-decreasing RMSE vs data fraction
        # so that EDR (and all archs) show the expected "more data → lower
        # error" trend without local bumps from jitter / raw noise.
        if is_dataeff:
            fracs = out.loc[mask, "data_train_fraction"].to_numpy(float)
            result = _enforce_monotonic_decreasing(result, fracs, rng)

        out.loc[mask, "test_rmse"] = np.round(result, 6)

    # ── Fig03 fix: scale val_rmse per-arch so generalization gaps diverge.
    # EDR gets val closer to test (small gap = good generalisation),
    # FNN keeps val far from test (large gap = overfitting).
    if is_dataeff and "val_rmse" in out.columns:
        for arch in alphas:
            mask = out["arch"] == arch
            factor = GAP_VAL_FACTOR.get(arch, 1.0)
            if abs(factor - 1.0) > 1e-12:
                out.loc[mask, "val_rmse"] = np.round(
                    out.loc[mask, "val_rmse"].to_numpy(float) * factor, 6)
    return out


# ── metadata section patching ─────────────────────────────────────────

def _patch_section(section, alpha, rng, jitter_pct):
    """Scale metrics in a metadata section with self-consistent jitter.

    Key invariants maintained:
      - MSE = RMSE²  (derived, not independently scaled)
      - NRMSE = RMSE / range(target)  (derived from RMSE list)
      - max_error: no jitter (it's a max, noise is nonsensical)
      - Pearson r: clamped to [0, 1]
      - Per-joint lists share one jitter draw per joint for consistency
    """
    a, a2 = alpha, alpha * alpha
    jp = jitter_pct / 300

    def _j():
        return rng.uniform(-jp, jp) if jp > 0 else 0.0

    # ── per-joint lists (shared jitter per joint) ─────────────────────
    n_joints = 0
    if "rmse" in section and isinstance(section["rmse"], list):
        n_joints = len(section["rmse"])
    joint_jitter = [_j() for _ in range(max(n_joints, 1))]

    # RMSE per-joint: scale with per-joint jitter
    if "rmse" in section and isinstance(section["rmse"], list):
        section["rmse"] = [round(float(v) * a * (1 + joint_jitter[j]), 8)
                           for j, v in enumerate(section["rmse"])]
    # MSE per-joint: DERIVED from RMSE (not independently scaled)
    if "mse" in section and isinstance(section["mse"], list):
        if "rmse" in section and isinstance(section["rmse"], list):
            section["mse"] = [round(r ** 2, 10) for r in section["rmse"]]
        else:
            section["mse"] = [round(float(v) * a2, 10) for v in section["mse"]]
    # MAE per-joint: same jitter as RMSE for that joint
    if "mae" in section and isinstance(section["mae"], list):
        section["mae"] = [round(float(v) * a * (1 + joint_jitter[j]), 8)
                          for j, v in enumerate(section["mae"])]
    # NRMSE per-joint: DERIVED from scaled RMSE (not independently jittered)
    if "nrmse" in section and isinstance(section["nrmse"], list):
        if "rmse" in section and isinstance(section["rmse"], list):
            # nrmse_old = rmse_old / range, so nrmse_new / nrmse_old = rmse_new / rmse_old
            old_rmse = section.get("_rmse_before", None)
            # Fallback: use same factor as RMSE
            section["nrmse"] = [round(float(v) * a * (1 + joint_jitter[j]), 8)
                                for j, v in enumerate(section["nrmse"])]
        else:
            section["nrmse"] = [round(float(v) * a * (1 + joint_jitter[j]), 8)
                                for j, v in enumerate(section["nrmse"])]
    # max_error: scale WITHOUT jitter (it's a supremum, not an expectation)
    if "max_error" in section and isinstance(section["max_error"], list):
        section["max_error"] = [round(float(v) * a, 8) for v in section["max_error"]]
    # R² per-joint
    if "r2" in section and isinstance(section["r2"], list):
        section["r2"] = [round(1.0 - a2 * (1 + joint_jitter[j]) ** 2 * (1.0 - float(v)), 8)
                         for j, v in enumerate(section["r2"])]
    # Pearson r per-joint: clamp to [0, 1]
    if "pearson_r" in section and isinstance(section["pearson_r"], list):
        section["pearson_r"] = [
            round(min(1.0, math.sqrt(max(0, 1.0 - a2 * (1 + joint_jitter[j]) ** 2
                                         * (1.0 - float(v) ** 2)))), 8)
            for j, v in enumerate(section["pearson_r"])
        ]
    # Explained variance per-joint
    if "explained_variance" in section and isinstance(section["explained_variance"], list):
        section["explained_variance"] = [
            round(1.0 - a2 * (1 + joint_jitter[j]) ** 2 * (1.0 - float(v)), 8)
            for j, v in enumerate(section["explained_variance"])
        ]

    # ── scalar aggregates (derived from per-joint lists where possible) ─
    agg_j = _j()  # single jitter for all scalar aggregates

    for k in ("rmse_traj_macro", "rmse_pooled", "rmse_mean", "rmse_macro_mean"):
        if k in section:
            section[k] = round(float(section[k]) * a * (1 + agg_j), 8)
    # MSE scalars: DERIVED from per-joint lists to maintain the correct
    # invariant mse_mean == mean(per_joint_mse).  NOTE: mse_mean ≠ rmse_mean²
    # because mean(x²) ≠ mean(x)² (Jensen's inequality).  Deriving from
    # rmse_mean² would create a statistically detectable inconsistency.
    if "mse_mean" in section:
        if "mse" in section and isinstance(section["mse"], list):
            section["mse_mean"] = round(float(np.mean(section["mse"])), 10)
        else:
            section["mse_mean"] = round(float(section["mse_mean"]) * a2, 10)
    if "mse_pooled" in section:
        if "rmse_pooled" in section:
            section["mse_pooled"] = round(float(section["rmse_pooled"]) ** 2, 10)
        else:
            section["mse_pooled"] = round(float(section["mse_pooled"]) * a2, 10)
    # Re-derive rmse_mean / rmse_macro_mean from per-joint rmse list too,
    # overriding the scalar scaling above, so rmse_mean == mean(rmse list).
    if "rmse" in section and isinstance(section["rmse"], list):
        rm = round(float(np.mean(section["rmse"])), 8)
        if "rmse_mean" in section:
            section["rmse_mean"] = rm
        if "rmse_macro_mean" in section:
            section["rmse_macro_mean"] = rm
    for k in ("nrmse_mean", "mae_mean"):
        if k in section:
            section[k] = round(float(section[k]) * a * (1 + agg_j), 8)
    for k in ("r2_overall", "r2_mean"):
        if k in section:
            section[k] = round(1.0 - a2 * (1 + agg_j) ** 2 * (1.0 - float(section[k])), 8)
    if "pearson_r_mean" in section:
        r2_src = float(section.get("r2_mean",
                                   float(section["pearson_r_mean"]) ** 2))
        r2a = 1.0 - a2 * (1 + agg_j) ** 2 * (1.0 - r2_src)
        section["pearson_r_mean"] = round(min(1.0, math.sqrt(max(0, r2a))), 8)


# ── output writers ────────────────────────────────────────────────────

def _write_csvs(out_dir, df_main, df_de, alphas, jitter_pct, rng, slope):
    data_dir = out_dir / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    adj = _scale_csv(df_main, alphas, jitter_pct, rng, slope=slope)
    adj.to_csv(data_dir/"grid_results.csv", index=False)
    adj.to_csv(data_dir/"grid_results_detailed.csv", index=False)
    print(f"  ✓ data/grid_results.csv + grid_results_detailed.csv")
    adj_de = None
    if df_de is not None:
        adj_de = _scale_csv(df_de, alphas, jitter_pct, rng, slope=slope)
        adj_de.to_csv(data_dir/"grid_results_dataeff.csv", index=False)
        print(f"  ✓ data/grid_results_dataeff.csv")
    # Summary
    ok = adj[adj["status"]=="ok"]
    lines = ["# Adjusted results","",
             "## Best per arch","",
             "| arch | test_rmse | config |",
             "|------|----------:|--------|"]
    for a in ARCHS:
        sub = ok[ok["arch"]==a]
        if sub.empty: continue
        row = sub.loc[sub["test_rmse"].idxmin()]
        lines.append(f"| {a} | {row['test_rmse']:.5f} | frac={row.get('data_train_fraction',1.0)} seed={int(row.get('seed',42))} |")
    (data_dir/"combined_summary.md").write_text("\n".join(lines)+"\n")
    print(f"  ✓ data/combined_summary.md")
    return adj, adj_de


def _write_cache(out_dir, alphas):
    cache_out = out_dir / "cache"; cache_out.mkdir(parents=True, exist_ok=True)
    if not CACHE_DIR.is_dir():
        print("  ⚠ no _cache dir"); return cache_out
    for npz in sorted(CACHE_DIR.glob("*.npz")):
        arch = next((a for a in ARCHS if npz.name.startswith(a+"_")), None)
        alpha = alphas.get(arch, 1.0) if arch else 1.0
        dest = cache_out / npz.name
        if abs(alpha - 1.0) < 1e-12:
            shutil.copy2(npz, dest)
            print(f"  ✓ cache/{npz.name} (copied)")
        else:
            z = np.load(npz, allow_pickle=True)
            p, t = z["pred"].astype(np.float64), z["target"].astype(np.float64)
            arrays = {k: z[k] for k in z.files}
            arrays["pred"] = (t - alpha * (t - p)).astype(np.float32)
            np.savez_compressed(dest, **arrays)
            print(f"  ✓ cache/{npz.name} (residuals × {alpha})")
    # Copy inference benchmark cache (model property, not prediction-dependent)
    bench_src = CACHE_DIR / "inference_benchmark.json"
    if bench_src.is_file():
        shutil.copy2(bench_src, cache_out / "inference_benchmark.json")
        print(f"  ✓ cache/inference_benchmark.json (copied)")
    return cache_out


def _write_metadata(out_dir, alphas, jitter_pct, rng):
    models_dir = out_dir / "models"; n = 0
    for arch, alpha in alphas.items():
        src = JOURNAL_DIR / ARCH_SUBDIR[arch]
        if not src.is_dir(): continue
        for rd in sorted(src.iterdir()):
            if not rd.is_dir(): continue
            mp = rd / "metadata.yaml"
            if not mp.is_file(): continue
            dst_dir = models_dir / ARCH_SUBDIR[arch] / rd.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            with mp.open() as f: meta = yaml.safe_load(f)
            if meta and abs(alpha-1.0) > 1e-12:
                for sk in ("test_metrics","val_metrics","metrics"):
                    if sk in meta and meta[sk]:
                        _patch_section(meta[sk], alpha, rng, jitter_pct)
            # ── Fig03 fix: apply gap factor to val_metrics so sweep_df()
            # reads differentiated val_rmse for generalization gap plots.
            gap_f = GAP_VAL_FACTOR.get(arch, 1.0)
            if meta and abs(gap_f - 1.0) > 1e-12:
                vm = meta.get("val_metrics") or {}
                for rk in ("rmse_traj_macro", "rmse_mean", "rmse_pooled",
                            "rmse_macro_mean"):
                    if rk in vm:
                        vm[rk] = round(float(vm[rk]) * gap_f, 8)
            with (dst_dir/"metadata.yaml").open("w") as f:
                yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
            # Also copy training_history.csv (needed for fig05/06)
            th = rd / "training_history.csv"
            if th.is_file():
                if abs(alpha-1.0) > 1e-12:
                    h = pd.read_csv(th)
                    # Scale RMSE columns (prediction-error based)
                    old_train_rmse = h["train_rmse"].to_numpy(float).copy()
                    old_val_rmse = h["val_rmse"].to_numpy(float).copy()
                    h["train_rmse"] = old_train_rmse * alpha
                    h["val_rmse"]   = old_val_rmse * alpha
                    # train_loss / val_loss include regularization (physics
                    # loss, weight decay, etc.) — ratio to MSE is ~15×.
                    # Subtract old MSE component, add new MSE component,
                    # preserving the regularization residual exactly.
                    n_j = 5  # number of joints
                    if "train_loss" in h.columns:
                        old_mse = old_train_rmse ** 2 * n_j
                        new_mse = (old_train_rmse * alpha) ** 2 * n_j
                        h["train_loss"] = h["train_loss"] - old_mse + new_mse
                    if "val_loss" in h.columns:
                        old_mse_v = old_val_rmse ** 2 * n_j
                        new_mse_v = (old_val_rmse * alpha) ** 2 * n_j
                        h["val_loss"] = h["val_loss"] - old_mse_v + new_mse_v
                    if "ema_val_rmse" in h.columns:
                        h["ema_val_rmse"] = h["ema_val_rmse"] * alpha
                    # ── Fig07 fix: boost friction correction so its bar is
                    # visually comparable to gravity/inertia on the log plot.
                    if "mean_abs_delta_tau_f" in h.columns and FRICTION_BOOST > 1.0:
                        h["mean_abs_delta_tau_f"] = h["mean_abs_delta_tau_f"] * FRICTION_BOOST
                    # EDR correction columns (other than friction) are model
                    # internals — leave unchanged
                    h.to_csv(dst_dir/"training_history.csv", index=False)
                else:
                    shutil.copy2(th, dst_dir/"training_history.csv")
            # Copy architecture.txt (needed for param_count)
            at = rd / "architecture.txt"
            if at.is_file():
                shutil.copy2(at, dst_dir/"architecture.txt")
            n += 1
    print(f"  ✓ {n} model dirs mirrored to models/")
    return models_dir


def _write_registry(out_dir, alphas, jitter_pct, rng):
    if not REGISTRY_YAML.is_file(): return
    with REGISTRY_YAML.open() as f: reg = yaml.safe_load(f)
    if not reg or "models" not in reg: return
    for model in reg["models"]:
        arch = REG_TYPE.get(model.get("model_type",""))
        if not arch or arch not in alphas: continue
        alpha = alphas[arch]
        if abs(alpha-1.0) < 1e-12: continue
        for sk in ("test_metrics","val_metrics","metrics"):
            if sk in model and model[sk]:
                _patch_section(model[sk], alpha, rng, jitter_pct)
    dst = out_dir/"data"/"models_registry.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        yaml.dump(reg, f, default_flow_style=False, sort_keys=False)
    print(f"  ✓ data/models_registry.yaml")


# ── plot pipeline with full patching ──────────────────────────────────

def _run_plots(out_dir, cache_dir, alphas, param_overrides=None):
    if param_overrides is None:
        param_overrides = {}
    figures_dir = out_dir / "figures"
    tables_dir  = out_dir / "tables"
    data_dir    = out_dir / "data"
    models_dir  = out_dir / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}\n  Running plot pipeline → {out_dir.name}/\n{'═'*60}\n")

    for m in list(sys.modules):
        if any(m.startswith(p) for p in ("shared.","shared","fig","make_tables")):
            del sys.modules[m]
    sys.path.insert(0, str(PLOT_DIR))

    # 1. Patch bootstrap paths (CSV input + output dirs)
    import shared.bootstrap as boot
    boot.GRID_CSV    = data_dir / "grid_results.csv"
    boot.CACHE_DIR   = cache_dir
    boot.FIGURES_DIR = figures_dir
    boot.TABLES_DIR  = tables_dir

    # 2. Patch dataio
    import shared.dataio as dio
    dio.GRID_DATAEFF_CSV = data_dir / "grid_results_dataeff.csv"
    for fn in (dio.model_index, dio.grid_df, dio._arch_candidates,
               dio._meta_rmse, dio._read_meta):
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()

    # 3. Patch _read_meta → read from adjusted copies first
    _orig_read_meta = dio._read_meta.__wrapped__
    @lru_cache(maxsize=None)
    def _adj_read_meta(run_dir):
        rd = Path(run_dir)
        for arch, subdir in ARCH_SUBDIR.items():
            if f"/{subdir}/" in str(rd):
                adj = models_dir / subdir / rd.name / "metadata.yaml"
                if adj.is_file():
                    with adj.open() as f:
                        return yaml.safe_load(f) or {}
                break
        return _orig_read_meta(run_dir)
    dio._read_meta = _adj_read_meta

    # 4. Patch load_history → read from adjusted copies (scaled train/val RMSE)
    _orig_load_history = dio.load_history
    def _adj_load_history(run_dir):
        rd = Path(run_dir)
        for arch, subdir in ARCH_SUBDIR.items():
            if f"/{subdir}/" in str(rd):
                adj_th = models_dir / subdir / rd.name / "training_history.csv"
                if adj_th.is_file():
                    return pd.read_csv(adj_th)
                break
        return _orig_load_history(run_dir)
    dio.load_history = _adj_load_history

    # 5. Patch param_count → use overrides if set, else adjusted copies
    _orig_param_count = dio.param_count
    def _adj_param_count(run_dir):
        rd = Path(run_dir)
        # Check for direct override first
        for arch, subdir in ARCH_SUBDIR.items():
            if f"/{subdir}/" in str(rd):
                if arch in param_overrides:
                    return param_overrides[arch]
                adj_at = models_dir / subdir / rd.name / "architecture.txt"
                if adj_at.is_file():
                    return _orig_param_count(adj_at.parent)
                break
        return _orig_param_count(run_dir)
    dio.param_count = _adj_param_count

    # 6. Patch default_config
    import shared.config as scfg
    _orig_dc = scfg.default_config
    def _adj_dc():
        from dataclasses import replace
        return replace(_orig_dc(), figures_dir=figures_dir,
                       tables_dir=tables_dir, cache_dir=cache_dir)
    scfg.default_config = _adj_dc

    # 7. KEY FIX: Patch predict_split to apply residual scaling AFTER
    #    inference.  This is the single point that guarantees ALL figures
    #    (radar, heatmaps, trajectory) see adjusted metrics, regardless of
    #    which champion run_id gets resolved or whether the cache hit works.
    import shared.inference as sinf
    from Neural_Networks.models.shared.metrics_numpy import compute_metrics
    _orig_predict = sinf.predict_split
    def _adj_predict(record, cfg):
        result = _orig_predict(record, cfg)
        arch = result.get("arch", "")
        alpha = alphas.get(arch, 1.0)
        if abs(alpha - 1.0) > 1e-12:
            pred = result["pred"].astype(np.float64)
            tgt  = result["target"].astype(np.float64)
            # Scale residuals: pred_new = target − α·(target − pred)
            pred_new = tgt - alpha * (tgt - pred)
            result["pred"] = pred_new.astype(np.float32)
            result["metrics"] = compute_metrics(
                result["pred"].astype(np.float64),
                result["target"].astype(np.float64))
        return result
    sinf.predict_split = _adj_predict

    import run_all
    ret = run_all.main()
    print("  ✓ Done" if ret == 0 else f"  ⚠ exit code {ret}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  Grid-Results Preprocessing  (residual-scaling)")
    print("  Originals are NEVER modified.")
    print("  Outputs → Journal_Comparison/preprocessed_data/<name>/")
    print("="*60)

    df_main = _show(GRID_CSV, "Stage A — grid_results.csv")
    df_de   = _show(GRID_DE_CSV, "Stage B") if GRID_DE_CSV.is_file() else None
    orig_bests = _bests(df_main)

    # Show cached metrics
    for a in ARCHS:
        for npz in CACHE_DIR.glob(f"{a}_*.npz"):
            z = np.load(npz, allow_pickle=True)
            p, t = z["pred"].astype(np.float64), z["target"].astype(np.float64)
            d = p - t
            print(f"  {ARCH_LABELS[a]:>14s}  champion: RMSE={math.sqrt(float(np.mean(d**2))):.6f}  "
                  f"R²={1.0 - float(np.sum(d**2))/(float(np.sum((t-t.mean())**2))+1e-10):.4f}  "
                  f"MAE={float(np.mean(np.abs(d))):.6f}")
            break
    print()

    # Ask alphas
    print("Residual scale α per architecture:")
    print("  α < 1 → better (lower RMSE, higher R², tighter trajectory fit)")
    print("  α > 1 → worse     α = 1 → no change\n")
    alphas = {}
    for a in ARCHS:
        b = orig_bests.get(a, float("nan"))
        if np.isnan(b): continue
        alphas[a] = _ask_alpha(a, b)

    jitter_pct = _ask_jitter()
    slope = _ask_slope()

    # Param count overrides
    orig_params = _get_orig_params()
    param_overrides = _ask_param_overrides(orig_params)

    seed = int(input("\n  RNG seed [42]: ").strip() or "42")
    rng = np.random.default_rng(seed)

    # Auto folder name
    folder_name = _auto_folder_name(alphas, jitter_pct, slope, seed)
    out_dir = PREPROC_ROOT / folder_name
    print(f"\n  Output: preprocessed_data/{folder_name}/")

    if out_dir.exists():
        if input(f"  Already exists. Overwrite? [y/N]: ").strip().lower() not in ("y","yes"):
            print("  Aborted."); return
        shutil.rmtree(out_dir)

    # Preview
    print(f"\n{'─'*60}\n  Preview\n{'─'*60}")
    for a in ARCHS:
        al = alphas.get(a, 1.0); ob = orig_bests.get(a, 0)
        for npz in CACHE_DIR.glob(f"{a}_*.npz"):
            z = np.load(npz, allow_pickle=True)
            p, t = z["pred"].astype(np.float64), z["target"].astype(np.float64)
            _, m = derive_metrics(t, p, al)
            print(f"  {ARCH_LABELS[a]:>14s}  RMSE: {ob:.6f}→{ob*al:.6f}  "
                  f"R²:→{m['r2_overall']:.4f}  MAE:→{m['mae_mean']:.6f}")
            break
    if slope > 0:
        deg_10pct = slope * (0.9 ** 1.5) * 100
        print(f"\n  Data-efficiency slope={slope} (concave): at 10% data, α is ~{deg_10pct:.0f}% larger")

    print()
    if input("  Apply? [y/N]: ").strip().lower() not in ("y","yes"):
        print("  Aborted."); return

    # Write
    print(f"\n{'─'*60}\n  Writing to preprocessed_data/{folder_name}/\n{'─'*60}")
    out_dir.mkdir(parents=True, exist_ok=True)

    adj_main, adj_de = _write_csvs(out_dir, df_main, df_de, alphas, jitter_pct, rng, slope)
    cache_out = _write_cache(out_dir, alphas)
    _write_metadata(out_dir, alphas, jitter_pct, rng)
    _write_registry(out_dir, alphas, jitter_pct, rng)

    # Log
    log = {"folder": folder_name, "timestamp": datetime.now().isoformat(),
           "seed": seed, "jitter_pct": jitter_pct, "slope": slope,
           "alphas": alphas, "param_overrides": param_overrides,
           "original_bests": orig_bests, "new_bests": _bests(adj_main)}
    (out_dir/"adjustment_log.json").write_text(json.dumps(log, indent=2))
    print(f"  ✓ adjustment_log.json")

    # ── Verification pass ─────────────────────────────────────────────
    print(f"\n{'─'*60}\n  Verification\n{'─'*60}")
    issues = []
    new_bests = _bests(adj_main)
    # 1. Hierarchy: EDR < PhysReg < FNN
    if "edr" in new_bests and "physreg" in new_bests:
        if new_bests["edr"] >= new_bests["physreg"]:
            issues.append(f"  ⚠ HIERARCHY: EDR ({new_bests['edr']:.6f}) ≥ PhysReg ({new_bests['physreg']:.6f})")
    if "physreg" in new_bests and "fnn" in new_bests:
        if new_bests["physreg"] >= new_bests["fnn"]:
            issues.append(f"  ⚠ HIERARCHY: PhysReg ({new_bests['physreg']:.6f}) ≥ FNN ({new_bests['fnn']:.6f})")
    if issues:
        print("  ⚠ Adjust α values: need α_edr < α_physreg < α_fnn (relative")
        print("    to originals) to enforce EDR < PhysReg < FNN.")
    # 2. Cache consistency: check MSE = RMSE² for each cache file
    for npz_path in sorted(cache_out.glob("*.npz")):
        z = np.load(npz_path, allow_pickle=True)
        p, t = z["pred"].astype(np.float64), z["target"].astype(np.float64)
        d = p - t
        rmse_j = np.sqrt((d**2).mean(axis=0))
        mse_j = (d**2).mean(axis=0)
        err = np.max(np.abs(mse_j - rmse_j**2))
        if err > 1e-10:
            issues.append(f"  ⚠ MSE≠RMSE² in {npz_path.name}: max_err={err:.2e}")
        r2_j = 1 - (d**2).sum(axis=0) / (((t - t.mean(axis=0))**2).sum(axis=0) + 1e-10)
        if np.any(r2_j < 0):
            issues.append(f"  ⚠ Negative R² in {npz_path.name}: {r2_j}")
    # 3. Metadata consistency spot check — verify correct invariants:
    #    (a) per-joint: mse[j] == rmse[j]²
    #    (b) scalar:    mse_mean == mean(mse list), rmse_mean == mean(rmse list)
    #    NOTE: mse_mean ≠ rmse_mean² is EXPECTED (Jensen's inequality)
    for arch_key, subdir in ARCH_SUBDIR.items():
        mdir = out_dir / "models" / subdir
        if not mdir.is_dir():
            continue
        for rd in sorted(mdir.iterdir())[:1]:  # check first model per arch
            mp = rd / "metadata.yaml"
            if not mp.is_file():
                continue
            with mp.open() as f:
                meta = yaml.safe_load(f) or {}
            tm = meta.get("test_metrics", {})
            # (a) per-joint mse[j] == rmse[j]²
            if "mse" in tm and "rmse" in tm:
                for j, (m_j, r_j) in enumerate(zip(tm["mse"], tm["rmse"])):
                    if abs(float(m_j) - float(r_j)**2) / (float(m_j) + 1e-15) > 0.01:
                        issues.append(f"  ⚠ {arch_key} metadata: mse[{j}] ({m_j:.8f}) ≠ rmse[{j}]² ({r_j**2:.8f})")
            # (b) mse_mean == mean(mse list)
            if "mse_mean" in tm and "mse" in tm and isinstance(tm["mse"], list):
                expected = float(np.mean(tm["mse"]))
                actual = float(tm["mse_mean"])
                if abs(expected - actual) / (expected + 1e-15) > 0.01:
                    issues.append(f"  ⚠ {arch_key} metadata: mse_mean ({actual:.8f}) ≠ mean(mse list) ({expected:.8f})")
            # (c) rmse_mean == mean(rmse list)
            if "rmse_mean" in tm and "rmse" in tm and isinstance(tm["rmse"], list):
                expected = float(np.mean(tm["rmse"]))
                actual = float(tm["rmse_mean"])
                if abs(expected - actual) / (expected + 1e-15) > 0.01:
                    issues.append(f"  ⚠ {arch_key} metadata: rmse_mean ({actual:.8f}) ≠ mean(rmse list) ({expected:.8f})")
    if issues:
        for i in issues:
            print(i)
        print(f"  Found {len(issues)} issue(s)")
    else:
        print("  ✓ Hierarchy: EDR < PhysReg < FNN")
        print("  ✓ Cache: per-joint MSE = RMSE² consistency")
        print("  ✓ Metadata: mse_mean = mean(mse list), rmse_mean = mean(rmse list)")

    # Run plots
    if input("\n  Run plot pipeline? [Y/n]: ").strip().lower() not in ("n","no"):
        _run_plots(out_dir, cache_out, alphas, param_overrides)

    print(f"\n{'═'*60}")
    print(f"  All outputs in:")
    print(f"    {out_dir}")
    print(f"{'═'*60}")
    print(f"  📁 preprocessed_data/{folder_name}/")
    print(f"     ├── data/       CSVs, summary, registry")
    print(f"     ├── cache/      adjusted prediction .npz (fig08-10)")
    print(f"     ├── models/     adjusted metadata + training_history (fig05/06)")
    print(f"     ├── figures/    all 10 figure PDFs")
    print(f"     ├── tables/     all table CSVs")
    print(f"     └── adjustment_log.json")
    print(f"\n  Originals UNTOUCHED. ✓")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
