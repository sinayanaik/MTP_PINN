# Journal_Comparison plotting suite

Visual-first comparison of **EDR** vs **FNN** vs **Physics-Regularized**
torque models: **11 single-purpose figures + 11 exhaustive CSV tables**. See
**`PLOTS_AND_TABLES.md`** for a per-artifact registry (what each shows, its
data source, how to read it, the takeaway, and the table behind it).

Full-data champion test RMSE (trajectory-macro, 87 596-sample test split):
**Physics-Reg. 0.0896 ≤ EDR 0.0902 ≤ FNN 0.0972 N·m**. EDR's value is
efficiency: ~best accuracy with ~40×/~5× fewer parameters than the baselines,
the best data-efficiency, and the smallest generalization gap. The grid CSVs
hold every trained run (EDR 72 / FNN 60 / PhysReg 48 at full data), so EDR's
grid distribution is the widest (it was swept over more HP combinations).

## Requirements

```bash
pip install SciencePlots          # REQUIRED (the theme is applied, not optional)
```
Plus the repo's normal env (torch, numpy, pandas, pyyaml, matplotlib, scipy).
Times New Roman must be installed for the intended typography.

## Run

```bash
cd Neural_Networks/Trained_Models/Journal_Comparison/plot_codes
python run_all.py                       # 11 figures + 11 tables
python run_all.py --only fig11          # subset of figures (skips tables)
python run_all.py --no-tables           # figures only
python make_tables.py                   # tables only
python fig02_data_efficiency_rmse.py    # any figure standalone
```

Common overrides (generic `--config-override KEY=VAL`):

```bash
--config-override champion_basis=val                   # champion by val (or train/global)
--config-override champion_full_data_only=false        # allow reduced-data champions
--config-override savgol_enabled=false                 # raw, unsmoothed lines
--config-override savgol_window=11 savgol_polyorder=3  # smoothing strength
--config-override trajectory_select=helix              # fig10 trajectory (or an int index)
--config-override dpi_save=600 fig_w=8                 # output tweaks
```

PDFs → `figures/`, CSVs → `tables/`. Champion predictions are computed once
(real inference over the test split) and cached in `_cache/`; delete it to force
recomputation (it auto-invalidates when a `model.pt` is newer than its cache).

## Per-figure styling (TWEAKABLES)

Every `figNN_*.py` opens with a `# ===== TWEAKABLES (edit me) =====` block of
plain module-level variables — figure size, dpi, per-arch colours, line/marker
sizes, font sizes, legend placement, axis labels, smoothing, and
figure-specific extras. Edit those to restyle that one figure; they feed
`CONFIG = replace(default_config(), ...)`. Shared defaults still live in
`shared/config.py`.

## Conventions

- **One chart per file/output.** No 2-in-1 panels.
- No titles, no `(a)/(b)` letters (enforced in `figio.save_pdf`); multi-panel
  identity is the axis label.
- **Non-bold Times New Roman**, uniform (STIX for math glyphs).
- One horizontal frameless legend strip centred above the plot.
- Baselines drawn first; EDR (red, heavier) on top — a focus choice.
- **Savitzky–Golay** smoothing is on by default and *replaces* the raw line in
  series/curve/trajectory figures; scatter/violin/heatmap are never smoothed.
  **All tables store raw, unsmoothed data.**
- Simple scalar data ⇒ CSV only (no redundant chart); richer data ⇒ chart +
  companion CSV. No two artifacts represent the same data the same way.
- PDF only (vector, 300 dpi). Every figure also has a raw-data CSV.

## Layout

- `shared/` — single source of truth: `config.PlotConfig` (sizes, colours,
  savgol, champion basis, trajectory selection, heatmap stops), `dataio` (the
  grid-CSV ⋈ on-disk-dir model index + champions by global/train/val/test),
  cached `inference`, `style` (SciencePlots + non-bold Times), `figio` (PDF),
  `tableio` (CSV), and `plotting` helpers (`top_legend`, `maybe_smooth`,
  `heatmap`, `per_traj_rmse`, `select_trajectory`).
- `figNN_*.py` — one self-contained figure each (with its TWEAKABLES block).
- `make_tables.py` — the 11 CSV tables.
