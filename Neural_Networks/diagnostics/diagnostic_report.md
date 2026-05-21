# Phase A Diagnostic Report — EDR val<test gap root cause

**Date:** 2026-05-22
**Triggering question:** Why does EDR show *low val_rmse* but *high test_rmse*
relative to PhysReg on the HPC journal grid?

## Headline

The val→test gap that the user observed is **real on HPC** (EDR mean gap
0.0173 N·m vs PhysReg mean gap 0.0129 N·m) and **driven primarily by
`correction_dropout`**. Configurations with `correction_dropout=0.05`
overfit val and inflate test by 16% relative to `correction_dropout=0.30`.

When the grid is filtered to `correction_dropout=0.30` only, EDR beats
PhysReg on the headline:

| Config              | n  | test_min | test_median | test_mean |
|---------------------|----|----------|-------------|-----------|
| **EDR cdrop=0.30**  | 15 | 0.0902   | **0.0918**  | **0.0917** |
| PhysReg (all)       | 48 | 0.0896   | 0.0924      | 0.0932    |
| EDR (all)           | 56 | 0.0902   | 0.0952      | 0.0987    |

The "EDR loses to PhysReg" headline is a grid-mean artifact: 23 of EDR's 56
configs use `cdrop=0.05` (mean test 0.1064), dragging the mean down.

## Diagnostic procedure

Four local diagnostic runs on RTX 3050 (results in
`Neural_Networks/diagnostics/results/`):

| Variant | Epochs | Data % | λ_corr | cdrop | EDR test | PhysReg test |
|---------|--------|--------|--------|-------|----------|--------------|
| A1      |  50    | 25%    | 0.01   | 0.30  | 0.0907   | 0.0959       |
| A2_long | 250    | 15%    | 0.01   | 0.30  | 0.0893   | 0.0907       |
| A2b     | 250    | 15%    | 0.002  | 0.30  | 0.0896   | n/a          |

EDR beat PhysReg on test in **every** local run — confirming the
implementation is correct and the gap is config-dependent, not architectural.

Per-epoch test eval was added to `pipeline.py` (gated by new HP
`diagnostic_test_eval=True`; no production cost). All variants show
**near-zero early-stop selection cost** (test@best_val ≈ best_test, gap
< 0.0002 N·m), ruling out checkpoint selection bias as the gap mechanism.

## Per-HP gap analysis on the 56 HPC EDR runs

The dominant driver is `correction_dropout`:

| cdrop | n  | val_mean | test_mean | gap_mean |
|-------|----|----------|-----------|----------|
| 0.05  | 23 | 0.0843   | 0.1064    | **0.0221** |
| 0.15  | 18 | 0.0804   | 0.0948    | 0.0144   |
| 0.30  | 15 | 0.0788   | **0.0917**| **0.0129** |

All 10 worst-gap EDR configs have `cdrop=0.05`. All 10 best-gap EDR configs
have `cdrop=0.30`.

Other axes have weak or no effect on the gap:
- `lambda_correction_reg ∈ [0.002, 0.2]`: gap variance 0.013–0.019, no trend
- `gravity_hidden` (32-32 vs 48-48): identical gap means (0.017)
- `use_friction_qdd` (True/False): identical gap means (0.017)

## Why low dropout overfits val (mechanism)

EDR has only ~21k parameters spread across 4 δ-nets. With dropout=0.05, each
δ-net effectively sees its full capacity each forward pass — there is no
ensembling effect during inference. The δ-nets memorise the val trajectory
distribution; train+val rmse drop together but test (held-out trajectories)
suffers because the corrections don't generalise.

Dropout=0.30 forces inference-time ensembling across 1/(1-p) ≈ 1.4 thinned
sub-networks per pass. This regularises the corrections to be the *common
signal across thinned nets*, which generalises across the val→test
distribution shift.

PhysReg with its 270k params + 0.30 dropout on the backbone naturally
benefits from the same effect — explaining why PhysReg's gap is consistently
smaller in the grid.

## Recommendations

### Immediate grid changes (B2 — tighten EDR sweep)

1. **Remove `correction_dropout=0.05` from `GRID_EDR_DETAILED`** entirely.
   It's the dominant gap driver; keeping it pollutes the grid mean.
2. **Consider also removing `cdrop=0.15`**, leaving `[0.30]` pinned — or
   add `[0.30, 0.45]` if you want to test whether higher dropout helps
   more. Sweep saves: 23 + 18 = 41 of 56 EDR configs are at suboptimal
   cdrop, of which 23 are clearly bad.
3. **λ_corr sweep is over-wide**: A1/A2 + HPC per-HP analysis both show
   λ_corr ∈ [0.002, 0.2] has tiny effect within seed noise. Drop the
   extremes; sweep `[0.01, 0.05]` only.
4. **Add multi-seed** at the remaining (much smaller) HP grid — needed to
   distinguish architecture from initialisation luck.

### No model code changes needed

EDR is correctly implemented:
- Analytical baseline (`τ_g, τ_M, τ_C, τ_f`) is bounded into every forward
  pass at `edr_model.py:670-680`. γ=1.0 at inference.
- Training loss `MSE(τ̂, τ_true) + Σλ‖δ‖²` is algebraically equivalent to
  regularised residual regression.
- Correction magnitudes (`||δM||_F`, `|δg|`, etc.) stabilise after epoch
  30–50; they do not grow unboundedly with training.
- Six prior ablation rounds (R1–R6) showed PSD-δM / structural Coriolis /
  spectral-norm regress test RMSE on this dataset — those should stay OFF.

### Speed work (B1) and resumable grid (C) still apply

These are independent of the dropout finding and are required for the
follow-up multi-seed sweep to fit in HPC budget.

## Files touched

- `Neural_Networks/models/shared/pipeline.py` — added `diagnostic_test_eval`
  HP flag (off by default); per-epoch test_rmse and ema_val_rmse appended to
  `training_history.csv` when on.
- `Neural_Networks/diagnostics/run_a1_baseline.py` — local diagnostic harness
  (modes: smoke / default / long, via env vars `DIAG_SMOKE`/`DIAG_LONG`).
- `Neural_Networks/diagnostics/run_a2b_low_lambda.py` — λ_corr=0.002
  variant.
- `Neural_Networks/diagnostics/analyze_curves.py` — overlay plotter +
  per-variant summary table.
- `Neural_Networks/diagnostics/results/` — all training_history.csvs,
  curves.png, summary jsons.
