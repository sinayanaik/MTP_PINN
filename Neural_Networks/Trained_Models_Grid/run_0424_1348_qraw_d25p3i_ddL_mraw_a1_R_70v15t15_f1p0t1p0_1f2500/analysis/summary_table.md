# Grid search — best run per architecture

Best = minimum **test** `rmse_pooled` within each `model_type`.

| Architecture | test RMSE (N·m) | test R2 overall | test MAE | run id |
|-------------|-----------------|----------------|----------|--------|
| Residual-Corr | 0.11503 | 0.87624 | 0.07630 | `ResidualCorrectionFNN_ep150_rmse0.11999_frac0.75_lr3e-4_wd0.01_do0.1_bs1024_hl25` |
| Physics-Reg | 0.13157 | 0.83809 | 0.08707 | `PhysicsRegularizedFNN_ep375_rmse0.13174_frac0.5_lr3e-4_wd0.01_do0.1_bs1024_hl256` |
| Black-Box | 0.14051 | 0.81534 | 0.10071 | `BlackBoxFNN_ep156_rmse0.14045_frac0.1_lr3e-4_wd0.01_do0.1_bs1024_hl256-512-256_2` |

## Test RMSE range across *all* runs in grid

- **Black-Box** (BlackBoxFNN): RMSE in [0.14051, 0.15778] N·m over 10 run(s); R2 in [0.76716, 0.81534].
- **Physics-Reg** (PhysicsRegularizedFNN): RMSE in [0.13041, 0.15439] N·m over 70 run(s); R2 in [0.77706, 0.84094].
- **Residual-Corr** (ResidualCorrectionFNN): RMSE in [0.11503, 0.12117] N·m over 50 run(s); R2 in [0.86268, 0.87624].
