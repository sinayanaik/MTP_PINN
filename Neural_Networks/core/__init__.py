"""
Neural_Networks.core
=====================
Core math and training logic — no Rich, no CLI, no Tkinter.

Exports:
  metrics.py  — compute_metrics, pooled_rmse_numpy, trajectory_mean_rmse_numpy
  builder.py  — build_model, build_optimizer, build_scheduler
  trainer.py  — train_epoch, eval_epoch, PhysicsWeightScheduler,
                 train_model (full pipeline for one model),
                 update_registry, save_comparison_plot
"""

from Neural_Networks.core.metrics import (       # noqa: F401
    compute_metrics,
    pooled_rmse_numpy,
    trajectory_mean_rmse_numpy,
)
from Neural_Networks.core.builder import (       # noqa: F401
    build_model,
    build_optimizer,
    build_scheduler,
)
from Neural_Networks.core.trainer import (       # noqa: F401
    PhysicsWeightScheduler,
    train_epoch,
    eval_epoch,
    train_model,
    update_registry,
    save_comparison_plot,
)
