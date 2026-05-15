"""Plot registry. The central runner iterates this dict in order;
the CLI uses `list(PLOTS)` for `--plot` argument validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import (
    training_dynamics,
    rmse_comparison,
    per_joint_heatmaps,
    mae_nrmse_comparison,
    edr_physics_corrections,
    topk_leaderboard,
    hp_pair_heatmaps,
    pareto_front,
    data_efficiency,
    physics_weight_impact,
    test_r2_distribution,
)


@dataclass(frozen=True)
class PlotSpec:
    fn: Callable[..., None]
    requires: Callable[[dict[str, Any]], bool] = field(
        default=lambda groups: True
    )


def _has_edr(groups: dict[str, Any]) -> bool:
    return bool(groups.get("EDR"))


PLOTS: dict[str, PlotSpec] = {
    "training_dynamics":       PlotSpec(training_dynamics.plot),
    "rmse_comparison":         PlotSpec(rmse_comparison.plot),
    "per_joint_heatmaps":      PlotSpec(per_joint_heatmaps.plot),
    "mae_nrmse_comparison":    PlotSpec(mae_nrmse_comparison.plot),
    "edr_physics_corrections": PlotSpec(edr_physics_corrections.plot, requires=_has_edr),
    "topk_leaderboard":        PlotSpec(topk_leaderboard.plot),
    "hp_pair_heatmaps":        PlotSpec(hp_pair_heatmaps.plot),
    "pareto_front":            PlotSpec(pareto_front.plot),
    "data_efficiency":         PlotSpec(data_efficiency.plot),
    "physics_weight_impact":   PlotSpec(physics_weight_impact.plot),
    "test_r2_distribution":    PlotSpec(test_r2_distribution.plot),
}
