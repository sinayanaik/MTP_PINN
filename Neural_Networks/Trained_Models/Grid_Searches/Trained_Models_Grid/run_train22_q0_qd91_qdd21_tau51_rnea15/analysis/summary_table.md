# Grid search — best run per architecture

Best = minimum **test** `rmse_pooled` within each `model_type`.

| Architecture | test RMSE (N·m) | test R2 overall | test MAE | run id |
|-------------|-----------------|----------------|----------|--------|
| Residual-Corr | 0.09666 | 0.91061 | 0.06480 | `ResidualCorrectionFNN_ep339_rmse0.10979_frac0.05_lr3e-4_wd0.05_do0.2_bs1024_hl12` |
| Physics-Reg | 0.09804 | 0.90803 | 0.06725 | `PhysicsRegularizedFNN_ep1075_rmse0.11026_frac0.05_lr3e-4_wd0.05_do0.2_bs1024_hl1` |
| Black-Box | 0.10566 | 0.89319 | 0.07193 | `BlackBoxFNN_ep300_rmse0.11712_frac1_lr3e-4_wd0.05_do0.2_bs1024_hl128-256-128_202` |

## Test RMSE range across *all* runs in grid

- **Black-Box** (BlackBoxFNN): RMSE in [0.10566, 0.11583] N·m over 12 run(s); R2 in [0.87163, 0.89319].
- **Physics-Reg** (PhysicsRegularizedFNN): RMSE in [0.09804, 0.10500] N·m over 72 run(s); R2 in [0.89452, 0.90803].
- **Residual-Corr** (ResidualCorrectionFNN): RMSE in [0.09666, 0.10112] N·m over 60 run(s); R2 in [0.90217, 0.91061].
