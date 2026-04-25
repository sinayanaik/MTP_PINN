# Grid search — best run per architecture

Best = minimum **test** `rmse_pooled` within each `model_type`.

| Architecture | test RMSE (N·m) | test R2 overall | test MAE | run id |
|-------------|-----------------|----------------|----------|--------|
| Physics-Reg | 0.10588 | 0.89290 | 0.07429 | `PhysicsRegularizedFNN_ep356_rmse0.11586_frac0.25_lr3e-4_wd0.01_do0.1_bs1024_hl25` |
| Residual-Corr | 0.10693 | 0.89077 | 0.07499 | `ResidualCorrectionFNN_ep350_rmse0.11750_frac1_lr3e-4_wd0.01_do0.1_bs1024_hl256-5` |
| Black-Box | 0.11953 | 0.86351 | 0.08457 | `BlackBoxFNN_ep350_rmse0.12865_frac1_lr3e-4_wd0.01_do0.1_bs1024_hl256-512-256_202` |

## Test RMSE range across *all* runs in grid

- **Black-Box** (BlackBoxFNN): RMSE in [0.11953, 0.13056] N·m over 12 run(s); R2 in [0.83718, 0.86351].
- **Physics-Reg** (PhysicsRegularizedFNN): RMSE in [0.10588, 0.11896] N·m over 120 run(s); R2 in [0.86482, 0.89290].
- **Residual-Corr** (ResidualCorrectionFNN): RMSE in [0.10693, 0.11219] N·m over 60 run(s); R2 in [0.87976, 0.89077].
