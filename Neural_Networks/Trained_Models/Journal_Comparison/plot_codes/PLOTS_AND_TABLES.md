# Plots & Tables registry

Every artifact, keyed by **filename**: what it shows, its data source, how to
read it, and the EDR-efficacy takeaway. Regenerate with `python run_all.py`
(figures → `figures/*.pdf`, tables → `tables/*.csv`).

**Models.** `EDR` (analytical dynamics + structured learned corrections,
~21 k params), `Physics-Reg.` (black-box net + physics loss, ~275 k), `FNN`
(plain MLP, ~270 k). Champion test RMSE (trajectory-macro, 87 596-sample test
split): EDR **0.0898** ≤ Physics-Reg. 0.0908 ≤ FNN 0.0998 N·m.

**Conventions.** One chart per file. No titles / `(a)(b)` letters (multi-panel
identity is the axis label). Non-bold Times New Roman. One horizontal
frameless legend strip above the plot. EDR is red, heavier, drawn on top.

**Savitzky–Golay.** On by default and *replacing* the raw line in every
series/curve/trajectory figure (fig02, fig03, fig06, fig07, fig08, fig11).
Disable: `python run_all.py --config-override savgol_enabled=false` (also
`savgol_window=`, `savgol_polyorder=`). Scatter / violin / heatmap figures are
never smoothed. **Tables always contain raw, unsmoothed data.**

**Round-1 → round-2 mapping.** old 01→fig01; old 02 split→fig02+fig03; old 03
split→fig04+fig05; old 04 split→fig06+fig07; old 05→fig08; old 06 (lollipop)
→fig09+fig10 (heatmaps); old 11 reworked→fig11. Old 07/08/09/10/12/13 removed.

---

## Figures

### `fig01_grid_rmse_distribution.pdf`
Violin + jittered points of test RMSE across all grid runs/arch; black bar =
median, star = champion (annotated). *Data:* `grid_results.csv`
(`status==ok`). *Read:* lower & tighter = better. *Takeaway:* EDR's cloud is
lowest and narrowest — most accurate and most consistent. *Table:*
`grid_summary.csv`, `grid_runs.csv`.

### `fig02_data_efficiency_rmse.pdf`
Test RMSE vs % training data, one line/arch (savgol). *Data:*
`dataio.sweep_df()`. *Read:* low & flat = data efficient. *Takeaway:* EDR
holds ~0.090 N·m at 10 % data; FNN degrades. *Table:* `data_efficiency.csv`.

### `fig03_generalization_gap_vs_fraction.pdf`
(test − val) RMSE vs % training data (savgol). *Data:* `sweep_df()`. *Read:*
near zero = generalizes. *Takeaway:* EDR's gap is lowest at every budget.
*Table:* `data_efficiency.csv`.

### `fig04_accuracy_vs_params.pdf`
Scatter: test RMSE vs parameter count (log x); champions starred. *Data:*
`registry_records()` + `param_count()`. *Read:* lower-left = better.
*Takeaway:* EDR is lower-left with ~13× fewer params. *Table:*
`cost_accuracy.csv`.

### `fig05_accuracy_vs_traintime.pdf`
Scatter: test RMSE vs training minutes; champions starred. *Data:*
`registry_records()`. *Read:* lower-left = better. *Takeaway:* EDR reaches
the best accuracy fastest. *Table:* `cost_accuracy.csv`.

### `fig06_train_val_curves.pdf`
Per-epoch train (dashed) vs val (solid) RMSE, 3 champions (savgol). *Data:*
champion `training_history.csv`. *Read:* curves together = healthy.
*Takeaway:* FNN over-fits; EDR's curves stay close. *Table:*
`training_curves.csv`.

### `fig07_overfitting_gap_vs_epoch.pdf`
(val − train) RMSE vs epoch, 3 champions (savgol). *Data:* champion histories.
*Read:* growing = over-fitting. *Takeaway:* EDR's gap stays small/flat.
*Table:* `training_curves.csv`.

### `fig08_edr_correction_evolution.pdf`
EDR |δg|, ‖δM‖_F, |δC·q̇|, |δτ_f| vs epoch, log y (savgol). *Data:* EDR
champion history delta columns. *Read:* which analytical term EDR corrects.
*Takeaway:* interpretability the baselines lack. *Table:*
`edr_correction_evolution.csv`.

### `fig09_per_joint_rmse_heatmap.pdf`
Architecture × joint RMSE heatmap; **green = best (low), red = worst**,
normalized globally over the whole matrix; cells annotated; colorbar
best↔worst. *Data:* champion predictions. *Read:* greener row = better.
*Takeaway:* EDR's row is the greenest overall. *Table:*
`per_joint_metrics.csv`.

### `fig10_per_joint_r2_heatmap.pdf`
Architecture × joint R² heatmap; **green = best (high), red = worst**, global
normalization, annotated. *Data:* champion predictions. *Read:* greener =
better. *Takeaway:* EDR's row is the greenest overall. *Table:*
`per_joint_metrics.csv`.

### `fig11_trajectory_tracking.pdf`
Five stacked joint panels; each overlays measured + FNN + Physics-Reg. + EDR
over one selected test trajectory (savgol). Trajectory chosen by
`trajectory_select` (None=auto / int index / geometry name), e.g.
`--config-override trajectory_select=helix`. *Data:* champion predictions.
*Read:* line on measured = good tracking. *Takeaway:* EDR tracks measured
most closely on every joint. *Table:* `trajectory_tracking.csv`.

---

## Tables (`tables/*.csv`, raw data)

- **`headline_metrics.csv`** — one row/champion: traj-macro & pooled RMSE,
  R²_overall, R²_mean, per-joint RMSE J1–J5, params, train_seconds,
  epochs_ran, worst_traj_rmse.
- **`grid_summary.csv`** — per arch: n_runs, best/mean/std/median test RMSE.
- **`grid_runs.csv`** — every grid run: n, architecture, status, test_rmse,
  elapsed_sec, data_train_fraction, seed.
- **`data_efficiency.csv`** — architecture, data_fraction_pct, test_rmse,
  val_rmse, gen_gap (3×10 rows).
- **`cost_accuracy.csv`** — architecture, run_id, params, train_seconds,
  train_minutes, test_rmse, is_champion (all registry models).
- **`training_curves.csv`** — long form: architecture, epoch, train_rmse,
  val_rmse, overfit_gap (the 3 champions).
- **`edr_correction_evolution.csv`** — epoch + the 4 EDR δ-term magnitudes.
- **`per_joint_metrics.csv`** — architecture, joint, rmse, r2, mae, nrmse
  (3×5 rows).
- **`trajectory_tracking.csv`** — geometry, sample, joint, measured,
  pred_fnn, pred_physreg, pred_edr for the selected trajectory.
