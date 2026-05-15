# Grid search — best run per architecture

Best = minimum **test** `rmse_pooled` within each `model_type`.

| Architecture | test RMSE (N·m) | test R2 overall | test MAE | run id |
|-------------|-----------------|----------------|----------|--------|
| Residual-Corr | 0.09668 | 0.91057 | 0.06208 | `ResidualCorrectionFNN_ep300_rmse0.10617_frac1_lr3e-4_wd0.05_do0.3_bs1024_hl128-2` |
| Physics-Reg | 0.10373 | 0.89704 | 0.06799 | `PhysicsRegularizedFNN_ep697_rmse0.10944_frac1_lr3e-4_wd0.05_do0.3_bs1024_hl128-2` |
| Black-Box | 0.11303 | 0.87777 | 0.07521 | `BlackBoxFNN_ep300_rmse0.12107_frac1_lr3e-4_wd0.05_do0.3_bs1024_hl128-256-128_202` |

## Test RMSE range across *all* runs in grid

- **Black-Box** (BlackBoxFNN): RMSE in [0.11303, 0.11303] N·m over 1 run(s); R2 in [0.87777, 0.87777].
- **Physics-Reg** (PhysicsRegularizedFNN): RMSE in [0.10373, 0.10373] N·m over 1 run(s); R2 in [0.89704, 0.89704].
- **Residual-Corr** (ResidualCorrectionFNN): RMSE in [0.09668, 0.09668] N·m over 1 run(s); R2 in [0.91057, 0.91057].
