"""Aggregator + visualiser for grid-search studies.

Reads ``Trained_Models_GridSearch/<study_name>/grid_log.jsonl`` and produces:

    ranking.csv          — every trial, sorted by rmse_pooled ascending
    leaderboard.csv      — best seed per (model, cell_id), sorted
    pivot_<model>.csv    — per-model pivot: rmse_mean over (axis_a × axis_b)
    plots/
        rmse_by_model.png                — best-per-model bar chart
        rmse_vs_<axis>_<model>.png       — line plot per α-active model
        heatmap_<model>_<ax1>_<ax2>.png  — heatmap of 2 strongest axes

Usage::

    python -m Neural_Networks.apps.grid_report

No CLI arguments — which study to aggregate and whether to skip plots are
read from Neural_Networks/config/grids/active.py (STUDY_NAME, REPORT_TARGET,
REPORT_NO_PLOTS).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

from Neural_Networks.apps.grid_search import GRID_OUTPUT_DIR


def resolve_study_dir(name_or_path: str) -> str:
    """Resolve to a concrete batch folder (the one holding grid_log.jsonl).
    Accepts:
      - absolute/relative path to a batch folder (e.g. batch_003_12perms_…)
      - absolute/relative path to a folder *containing* batches (→ newest inside)
      - bare batch name resolved under Trained_Models_GridSearch/
    """
    def _is_run_dir(p: str) -> bool:
        return os.path.isfile(os.path.join(p, "grid_log.jsonl")) \
            or os.path.isfile(os.path.join(p, "metadata.json"))

    def _newest_subdir(p: str) -> str | None:
        subs = [os.path.join(p, d) for d in os.listdir(p)
                if os.path.isdir(os.path.join(p, d))]
        subs = [s for s in subs if _is_run_dir(s)]
        if not subs:
            return None
        subs.sort(key=os.path.getmtime, reverse=True)
        return subs[0]

    candidates = []
    if os.path.isdir(name_or_path):
        candidates.append(os.path.abspath(name_or_path))
    candidates.append(os.path.join(GRID_OUTPUT_DIR, name_or_path))

    for c in candidates:
        if not os.path.isdir(c):
            continue
        if _is_run_dir(c):
            return os.path.abspath(c)
        newest = _newest_subdir(c)
        if newest:
            return os.path.abspath(newest)

    raise FileNotFoundError(
        f"Study not found (no grid_log.jsonl under): {candidates}"
    )


def load_log(study_dir: str) -> list[dict]:
    path = os.path.join(study_dir, "grid_log.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"grid_log.jsonl missing in {study_dir}")
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return [r for r in rows if "error" not in r]


def flatten(r: dict) -> dict:
    """Flatten one record: promote cell_hp keys to top-level columns."""
    flat = {k: v for k, v in r.items() if k != "cell_hp"}
    for k, v in (r.get("cell_hp") or {}).items():
        flat[f"hp_{k}"] = _stringify(v)
    return flat


def _stringify(v: Any) -> Any:
    """Lists (hidden_layers) → comma-joined string for CSV/grouping."""
    if isinstance(v, list):
        return "[" + ",".join(str(x) for x in v) + "]"
    return v


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def make_ranking(rows: list[dict]) -> list[dict]:
    out = [flatten(r) for r in rows]
    out.sort(key=lambda r: float(r.get("rmse_pooled", r.get("rmse_mean", 1e9))))
    return out


def make_leaderboard(rows: list[dict]) -> list[dict]:
    """Best seed per (model, cell_id).  Also attach mean/std across seeds."""
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["model"], r["cell_id"])].append(r)

    leader: list[dict] = []
    for (model, cid), group in by_cell.items():
        rmses = [float(g.get("rmse_pooled", g.get("rmse_mean", 0.0))) for g in group]
        best = min(group, key=lambda g: float(g.get("rmse_pooled",
                                                    g.get("rmse_mean", 1e9))))
        row = flatten(best)
        row["rmse_best"] = min(rmses)
        row["rmse_mean_over_seeds"] = mean(rmses)
        row["rmse_std_over_seeds"]  = pstdev(rmses) if len(rmses) > 1 else 0.0
        row["n_seeds"]              = len(group)
        leader.append(row)
    leader.sort(key=lambda r: r["rmse_best"])
    return leader


def make_pivots(rows: list[dict], study_dir: str) -> None:
    """Per-model pivots over every pair of axes."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    pivot_dir = os.path.join(study_dir, "pivots")
    os.makedirs(pivot_dir, exist_ok=True)
    for model, group in by_model.items():
        axes = _axes_of(group)
        if not axes:
            continue
        # One flat table: (seed-averaged rmse) × (each axis)
        stats = _seed_stats(group)
        write_csv(os.path.join(pivot_dir, f"{model}.csv"), stats)


def _axes_of(group: list[dict]) -> list[str]:
    if not group:
        return []
    sample_hp = group[0].get("cell_hp") or {}
    return sorted(sample_hp.keys())


def _seed_stats(group: list[dict]) -> list[dict]:
    """Aggregate rmse across seeds per cell_id."""
    by_cid: dict[str, list[dict]] = defaultdict(list)
    for r in group:
        by_cid[r["cell_id"]].append(r)
    out: list[dict] = []
    for cid, recs in by_cid.items():
        rmses = [float(r.get("rmse_pooled", r.get("rmse_mean", 0.0))) for r in recs]
        row = {"cell_id": cid, "n_seeds": len(recs),
               "rmse_best": min(rmses),
               "rmse_mean": mean(rmses),
               "rmse_std":  pstdev(rmses) if len(rmses) > 1 else 0.0}
        for k, v in (recs[0].get("cell_hp") or {}).items():
            row[f"hp_{k}"] = _stringify(v)
        out.append(row)
    out.sort(key=lambda r: r["rmse_best"])
    return out


# ---------------------------------------------------------------------------
# Plots (optional; soft-fails without matplotlib)
# ---------------------------------------------------------------------------

def make_plots(rows: list[dict], study_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plots skipped] matplotlib unavailable: {e}")
        return

    plots_dir = os.path.join(study_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- Bar: best trial per model, annotated with epoch + hp ------------
    # For each model we pick the single trial with the lowest rmse_pooled
    # (across every cell × seed) and plot it.  The bar label carries the
    # actual best_epoch so readers see where the reported RMSE was achieved,
    # and the hp cell is printed inside the bar.
    best_by_model: dict[str, dict] = {}
    for r in rows:
        m = r["model"]
        rmse = float(r.get("rmse_pooled", r.get("rmse_mean", 0.0)))
        if m not in best_by_model or rmse < best_by_model[m]["rmse"]:
            best_by_model[m] = {
                "rmse":        rmse,
                "best_epoch":  int(r.get("best_epoch", r.get("epochs_trained", 0))),
                "epochs_run":  int(r.get("epochs_trained", 0)),
                "hp":          r.get("cell_hp") or {},
                "seed":        int(r.get("seed", 0)),
            }

    models = sorted(best_by_model.keys(), key=lambda m: best_by_model[m]["rmse"])
    vals   = [best_by_model[m]["rmse"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_pos  = list(range(len(models)))
    bars   = ax.bar(x_pos, vals, width=0.62, color="#2a9d8f",
                    edgecolor="#1d6e62")

    # Epoch callout above every bar, hp cell inside
    y_top = max(vals) if vals else 1.0
    for i, (m, bar) in enumerate(zip(models, bars)):
        info = best_by_model[m]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_top * 0.01,
                f"ep{info['best_epoch']}/{info['epochs_run']}",
                ha="center", va="bottom", fontsize=9, color="#1d3557")
        hp_label = ", ".join(
            f"{k}={_stringify(v)}" for k, v in info["hp"].items()
        ) or "—"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                hp_label, ha="center", va="center", rotation=0,
                fontsize=7.5, color="white")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("rmse_pooled (val+test, N·m)")
    ax.set_title("Best trial per model — label: best_epoch/epochs_trained")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, y_top * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "rmse_by_model.png"), dpi=130)
    plt.close(fig)

    # --- Data-efficiency plot -------------------------------------------
    # If data_train_fraction was swept, show rmse vs fraction for each model.
    has_frac_axis = any(
        "data_train_fraction" in (r.get("cell_hp") or {}) for r in rows
    )
    if has_frac_axis:
        by_mf: dict[str, dict[float, float]] = defaultdict(dict)
        ep_mf: dict[str, dict[float, int]]   = defaultdict(dict)
        for r in rows:
            f = float((r.get("cell_hp") or {}).get("data_train_fraction", 1.0))
            m = r["model"]
            rmse = float(r.get("rmse_pooled", r.get("rmse_mean", 0.0)))
            # Keep the best rmse across other axes/seeds at this (model, fraction).
            if f not in by_mf[m] or rmse < by_mf[m][f]:
                by_mf[m][f] = rmse
                ep_mf[m][f] = int(r.get("best_epoch", r.get("epochs_trained", 0)))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for m in sorted(by_mf.keys()):
            fracs = sorted(by_mf[m].keys())
            ys    = [by_mf[m][f] for f in fracs]
            ax.plot(fracs, ys, "-o", label=m)
            # Annotate each point with the epoch that achieved it
            for f, y in zip(fracs, ys):
                ax.annotate(f"ep{ep_mf[m][f]}", (f, y),
                            textcoords="offset points", xytext=(0, 6),
                            fontsize=7, ha="center", color="#555")
        ax.set_xlabel("data_train_fraction")
        ax.set_ylabel("rmse_pooled (val+test, N·m)")
        ax.set_title("Data efficiency — best trial per (model × fraction)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "data_efficiency.png"), dpi=130)
        plt.close(fig)

    # --- Per-axis line plots (seed-averaged) for α-active models ----------
    for model in ("PhysicsRegularizedFNN", "EquationConstrainedPINNFNN"):
        group = [r for r in rows if r["model"] == model]
        if not group:
            continue
        axes = _axes_of(group)
        for axis in axes:
            _plot_marginal(group, axis, model,
                           os.path.join(plots_dir, f"rmse_vs_{axis}_{model}.png"))

    # --- Batch comparison plots ------------------------------------------
    _plot_rmse_boxplot(rows, plots_dir, plt)
    _plot_training_curves(rows, study_dir, plots_dir, plt)
    _plot_heatmaps(rows, plots_dir, plt)
    _plot_r2_radar(rows, plots_dir, plt)

    print(f"  plots → {plots_dir}")


def _plot_rmse_boxplot(rows: list[dict], plots_dir: str, plt) -> None:
    """Box plot showing RMSE distribution per model across all cells/seeds."""
    by_model: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(
            float(r.get("rmse_pooled", r.get("rmse_mean", 0.0)))
        )
    if not by_model:
        return
    models = sorted(by_model.keys(),
                    key=lambda m: sorted(by_model[m])[len(by_model[m]) // 2])
    data = [by_model[m] for m in models]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bp = ax.boxplot(data, patch_artist=True, labels=models)
    colors = plt.cm.Set2([i / max(1, len(models) - 1) for i in range(len(models))])
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("rmse_pooled (N·m)")
    ax.set_title("RMSE distribution per model (all cells × seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "rmse_boxplot.png"), dpi=130)
    plt.close(fig)


def _plot_training_curves(rows: list[dict], study_dir: str,
                          plots_dir: str, plt) -> None:
    """Overlay val_rmse training curves from each run's training_history.csv."""
    import numpy as np
    model_colors: dict[str, str] = {}
    palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6a4c93"]

    all_models = sorted({r["model"] for r in rows})
    for i, m in enumerate(all_models):
        model_colors[m] = palette[i % len(palette)]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    legend_done: set[str] = set()

    for r in rows:
        sd = r.get("save_dir", "")
        csv_path = os.path.join(sd, "training_history.csv") if sd else ""
        if not csv_path or not os.path.isfile(csv_path):
            continue
        try:
            epochs_col: list[int] = []
            vrmse_col: list[float] = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs_col.append(int(row["epoch"]))
                    vrmse_col.append(float(row["val_rmse"]))
        except Exception:
            continue
        if not epochs_col:
            continue
        m = r["model"]
        label = m if m not in legend_done else None
        legend_done.add(m)
        ax.plot(epochs_col, vrmse_col, color=model_colors.get(m, "#888"),
                alpha=0.45, linewidth=0.9, label=label)

    if not legend_done:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_rmse (N·m)")
    ax.set_title("Training curves — val RMSE across all grid runs")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=130)
    plt.close(fig)


def _plot_heatmaps(rows: list[dict], plots_dir: str, plt) -> None:
    """Per-model heatmap of RMSE over the two axes with the most unique values."""
    import numpy as np
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    for model, group in by_model.items():
        axes_keys = _axes_of(group)
        if len(axes_keys) < 2:
            continue
        # Pick the two axes with the most distinct values
        axis_uniques: list[tuple[str, int]] = []
        for ak in axes_keys:
            vals = {_stringify((r.get("cell_hp") or {}).get(ak)) for r in group}
            axis_uniques.append((ak, len(vals)))
        axis_uniques.sort(key=lambda x: -x[1])
        ax1_name, ax2_name = axis_uniques[0][0], axis_uniques[1][0]

        # Build 2D map: (ax1_val, ax2_val) → best rmse
        rmse_map: dict[tuple, float] = {}
        for r in group:
            hp = r.get("cell_hp") or {}
            k = (_stringify(hp.get(ax1_name)), _stringify(hp.get(ax2_name)))
            rmse = float(r.get("rmse_pooled", r.get("rmse_mean", 0.0)))
            if k not in rmse_map or rmse < rmse_map[k]:
                rmse_map[k] = rmse

        ax1_vals = sorted({k[0] for k in rmse_map}, key=lambda v: (isinstance(v, str), v))
        ax2_vals = sorted({k[1] for k in rmse_map}, key=lambda v: (isinstance(v, str), v))
        if len(ax1_vals) < 2 or len(ax2_vals) < 2:
            continue

        grid = np.full((len(ax2_vals), len(ax1_vals)), np.nan)
        for (v1, v2), rmse in rmse_map.items():
            i = ax1_vals.index(v1)
            j = ax2_vals.index(v2)
            grid[j, i] = rmse

        fig, ax = plt.subplots(figsize=(max(6, len(ax1_vals) * 1.2),
                                        max(4, len(ax2_vals) * 0.8)))
        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd_r")
        ax.set_xticks(range(len(ax1_vals)))
        ax.set_xticklabels([str(v) for v in ax1_vals], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(ax2_vals)))
        ax.set_yticklabels([str(v) for v in ax2_vals], fontsize=8)
        ax.set_xlabel(ax1_name)
        ax.set_ylabel(ax2_name)
        ax.set_title(f"{model}: best RMSE heatmap")
        for j in range(len(ax2_vals)):
            for i in range(len(ax1_vals)):
                if not np.isnan(grid[j, i]):
                    ax.text(i, j, f"{grid[j, i]:.4f}", ha="center", va="center",
                            fontsize=7, color="black")
        fig.colorbar(im, ax=ax, label="rmse_pooled (N·m)", shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"heatmap_{model}.png"), dpi=130)
        plt.close(fig)


def _plot_r2_radar(rows: list[dict], plots_dir: str, plt) -> None:
    """Radar chart comparing per-joint R2 for each model's best trial."""
    import numpy as np

    # For each model, find the trial with the best rmse and read per-joint r2
    # from its save_dir/metadata.yaml
    best_by_model: dict[str, dict] = {}
    for r in rows:
        m = r["model"]
        rmse = float(r.get("rmse_pooled", r.get("rmse_mean", 0.0)))
        if m not in best_by_model or rmse < best_by_model[m]["rmse"]:
            best_by_model[m] = {"rmse": rmse, "save_dir": r.get("save_dir", "")}

    # Try loading per-joint R2 from metadata.yaml in each best run dir
    model_r2: dict[str, list[float]] = {}
    for m, info in best_by_model.items():
        sd = info.get("save_dir", "")
        if not sd:
            continue
        for meta_name in ("metadata.yaml",):
            meta_path = os.path.join(sd, meta_name)
            if not os.path.isfile(meta_path):
                continue
            try:
                import yaml
                with open(meta_path) as f:
                    md = yaml.safe_load(f) or {}
                r2_list = (md.get("metrics") or {}).get("r2")
                if r2_list and len(r2_list) == 5:
                    model_r2[m] = [float(x) for x in r2_list]
            except Exception:
                pass

    if len(model_r2) < 2:
        return

    joint_names = ["J1 yaw", "J2 shoulder", "J3 elbow", "J4 wrist", "J5 wrist roll"]
    n_joints = 5
    angles = np.linspace(0, 2 * np.pi, n_joints, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6a4c93"]
    for i, (m, r2) in enumerate(sorted(model_r2.items())):
        vals = r2 + r2[:1]
        ax.plot(angles, vals, "-o", linewidth=1.5, markersize=4,
                label=m, color=palette[i % len(palette)])
        ax.fill(angles, vals, alpha=0.08, color=palette[i % len(palette)])
    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], joint_names, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-joint R² — best trial per model", y=1.08)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "r2_radar.png"), dpi=130)
    plt.close(fig)


def _plot_marginal(group: list[dict], axis: str, model: str, out_path: str) -> None:
    import matplotlib.pyplot as plt
    buckets: dict[Any, list[float]] = defaultdict(list)
    for r in group:
        v = _stringify((r.get("cell_hp") or {}).get(axis))
        buckets[v].append(float(r.get("rmse_pooled", r.get("rmse_mean", 0.0))))
    if len(buckets) < 2:
        return
    xs = sorted(buckets.keys(), key=lambda k: (isinstance(k, str), k))
    means = [mean(buckets[x]) for x in xs]
    bests = [min(buckets[x])  for x in xs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(xs)), means, "-o", label="mean", color="#264653")
    ax.plot(range(len(xs)), bests, "--s", label="best", color="#e76f51")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(x) for x in xs], rotation=20, ha="right")
    ax.set_xlabel(axis)
    ax.set_ylabel("rmse_pooled")
    ax.set_title(f"{model}: rmse vs {axis} (marginalised over other axes)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(study_name: str, no_plots: bool = False) -> None:
    study_dir = resolve_study_dir(study_name)
    print(f"Study: {study_dir}")
    rows = load_log(study_dir)
    if not rows:
        print("  no completed trials yet.")
        return
    print(f"  {len(rows)} completed trials")

    ranking = make_ranking(rows)
    write_csv(os.path.join(study_dir, "ranking.csv"), ranking)
    print(f"  ranking → {os.path.join(study_dir, 'ranking.csv')}")

    leaderboard = make_leaderboard(rows)
    write_csv(os.path.join(study_dir, "leaderboard.csv"), leaderboard)
    print(f"  leaderboard → {os.path.join(study_dir, 'leaderboard.csv')}")

    make_pivots(rows, study_dir)
    print(f"  pivots → {os.path.join(study_dir, 'pivots/')}")

    print("\nTop 10 (model, cell, rmse_best):")
    for row in leaderboard[:10]:
        hp_str = ", ".join(f"{k[3:]}={v}" for k, v in row.items()
                           if k.startswith("hp_"))
        print(f"  {row['model']:30s}  cell={row['cell_id']}  "
              f"rmse={row['rmse_best']:.5f}  seeds={row['n_seeds']}  [{hp_str}]")

    if not no_plots:
        make_plots(rows, study_dir)


def main() -> None:
    """Load Neural_Networks.config.grids.active and aggregate.  No CLI args."""
    import importlib
    mod = importlib.import_module("Neural_Networks.config.grids.active")
    target   = getattr(mod, "REPORT_TARGET", None)
    no_plots = bool(getattr(mod, "REPORT_NO_PLOTS", False))

    # REPORT_TARGET can be: absolute path, a batch folder name under
    # Trained_Models_GridSearch/, or None (→ newest batch).
    if target:
        study_arg = target if (os.path.isabs(target) or os.path.isdir(target)) \
            else os.path.join(GRID_OUTPUT_DIR, target)
    else:
        study_arg = GRID_OUTPUT_DIR          # → newest batch_* folder

    run(study_arg, no_plots=no_plots)


if __name__ == "__main__":
    sys.exit(main())
