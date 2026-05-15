# Grid search — best run per architecture

Best = minimum **test** `rmse_pooled` within each `model_type`.

| Architecture | test RMSE (N·m) | test R2 overall | test MAE | run id |
|-------------|-----------------|----------------|----------|--------|
| Residual-Corr | 0.09700 | 0.90997 | 0.06324 | `ResidualCorrectionFNN_ep1055_rmse0.10785_frac0.05_lr3e-4_wd0.05_do0.2_bs1024_hl1` |
| Physics-Reg | 0.09883 | 0.90654 | 0.06686 | `PhysicsRegularizedFNN_ep1519_rmse0.11005_frac0.05_lr3e-4_wd0.05_do0.2_bs1024_hl1` |
| Black-Box | 0.10736 | 0.88972 | 0.07230 | `BlackBoxFNN_ep300_rmse0.11818_frac1_lr3e-4_wd0.05_do0.2_bs1024_hl128-256-128_202` |

## Test RMSE range across *all* runs in grid

- **Black-Box** (BlackBoxFNN): RMSE in [0.10736, 0.10736] N·m over 1 run(s); R2 in [0.88972, 0.88972].
- **Physics-Reg** (PhysicsRegularizedFNN): RMSE in [0.09883, 0.09944] N·m over 2 run(s); R2 in [0.90539, 0.90654].
- **Residual-Corr** (ResidualCorrectionFNN): RMSE in [0.09700, 0.09819] N·m over 2 run(s); R2 in [0.90775, 0.90997].
