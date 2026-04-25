# Grid search — best run per architecture

Best = minimum **test** `rmse_pooled` within each `model_type`.

| Architecture | test RMSE (N·m) | test R2 overall | test MAE | run id |
|-------------|-----------------|----------------|----------|--------|
| Physics-Reg | 0.07928 | 0.92799 | 0.04579 | `PhysicsRegularizedFNN_ep140_rmse0.08266_frac0.5_lr3e-4_wd0.01_do0.1_bs1024_hl256` |
| Residual-Corr | 0.08153 | 0.92384 | 0.04433 | `ResidualCorrectionFNN_ep278_rmse0.08320_frac1_lr3e-4_wd0.01_do0.1_bs1024_hl256-5` |
| Black-Box | 0.08303 | 0.92102 | 0.04639 | `BlackBoxFNN_ep130_rmse0.08808_frac0.5_lr3e-4_wd0.01_do0.1_bs1024_hl256-512-256_2` |

## Test RMSE range across *all* runs in grid

- **Black-Box** (BlackBoxFNN): RMSE in [0.08303, 0.15778] N·m over 30 run(s); R2 in [0.76716, 0.92102].
- **Physics-Reg** (PhysicsRegularizedFNN): RMSE in [0.07928, 0.15439] N·m over 230 run(s); R2 in [0.77706, 0.92799].
- **Residual-Corr** (ResidualCorrectionFNN): RMSE in [0.08153, 0.12117] N·m over 134 run(s); R2 in [0.86268, 0.92384].
