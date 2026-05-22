# Combined run — Stage A (HP grid) → Stage B (data-efficiency)

- Stage A: OK=16  Skip=108  Fail=0
- Stage B: OK=27  Skip=3  Fail=0

## Stage-A winning config per arch (min test_rmse)

| arch | test_rmse @ Stage A | config |
|------|--------------------:|--------|
| edr | 0.09048 | frac=1.0 seed=42 |
| fnn | 0.09717 | frac=1.0 seed=42 |
| physreg | 0.08956 | frac=1.0 seed=42 pw=1.0 |

## Headline (multi-seed)

The Stage-B rows with `data_train_fraction == 1.0` (seeds 42, 1, 2) ARE the multi-seed headline — see `grid_results_dataeff.csv` for the per-seed test RMSE and the plot suite's `sweep_df` for mean ± std.

Artifacts: Stage A → `grid_results_detailed.csv` (+ legacy `grid_results.csv` mirror); Stage B → `grid_results_dataeff.csv`.
