# Journal_Comparison plotting suite

Visual-first comparison of **EDR** vs **FNN** vs **Physics-Regularized**
torque models: **11 single-purpose figures + 9 exhaustive CSV tables**. See
**`PLOTS_AND_TABLES.md`** for a per-artifact registry (what each shows, its
data source, how to read it, the EDR takeaway, and the table behind it).

## Requirements

```bash
conda install -c conda-forge scienceplots   # optional; degrades gracefully
```
Plus the repo's normal env (torch, numpy, pandas, pyyaml, matplotlib, scipy).

## Run

```bash
cd Neural_Networks/Trained_Models/Journal_Comparison/plot_codes
python run_all.py                       # 11 figures + 9 tables
python run_all.py --only fig11          # subset of figures (skips tables)
python run_all.py --no-tables           # figures only
python make_tables.py                   # tables only
python fig02_data_efficiency_rmse.py    # any figure standalone
```

Common overrides (generic `--config-override KEY=VAL`):

```bash
--config-override savgol_enabled=false                 # raw, unsmoothed lines
--config-override savgol_window=11 savgol_polyorder=3  # smoothing strength
--config-override trajectory_select=helix              # fig11 trajectory (or an int index)
--config-override dpi_save=600 fig_w=8                 # output tweaks
```

PDFs → `figures/`, CSVs → `tables/`. Champion predictions are computed once
(real inference over the 87 596-sample test split) and cached in `_cache/`;
delete it to force recomputation.

## Conventions

- **One chart per file/output.** No 2-in-1 panels.
- No titles, no `(a)/(b)` letters (enforced in `figio.save_pdf`); multi-panel
  identity is the axis label.
- Non-bold **Times New Roman**, uniform.
- One horizontal frameless legend strip centred above the plot.
- Baselines drawn first; EDR (red, heavier) on top.
- **Savitzky–Golay** smoothing is on by default and *replaces* the raw line
  in series/curve/trajectory figures; scatter/violin/heatmap are never
  smoothed. **All tables store raw, unsmoothed data.**
- PDF only (vector, 300 dpi). Every figure also has a raw-data CSV.

## Layout

- `shared/` — single source of truth: `config.PlotConfig` (sizes, colours,
  savgol, trajectory selection, heatmap stops), `dataio`, cached `inference`,
  `style`, `figio` (PDF), `tableio` (CSV), and `plotting` helpers
  (`top_legend`, `maybe_smooth`, `heatmap`, `select_trajectory`).
- `figNN_*.py` — one self-contained figure each.
- `make_tables.py` — the 9 CSV tables.
