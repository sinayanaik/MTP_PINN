"""Fig 10 — Top-K leaderboard table per architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ..config import _ARCH_HP_KEYS, _GRID_HP_KEYS_FNN
from ..io.records import arch_short_label, rmse_scalar, split_scalar
from ._common import hp_val_str, save_fig


def plot(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    *,
    top_k: int = 10,
    **_: Any,
) -> None:
    arch_order = [t for t in ["BlackBoxFNN", "PhysicsRegularizedFNN", "ResidualCorrectionFNN", "EDR"]
                  if t in groups]
    arch_order += [t for t in sorted(groups) if t not in arch_order]
    if not arch_order:
        return

    n_archs = len(arch_order)
    fig_h = max(5.0, top_k * 0.52 + 2.0) * n_archs
    fig, axes = plt.subplots(n_archs, 1, figsize=(22, fig_h))
    if n_archs == 1:
        axes = [axes]

    _HP_HEADER = {
        "hidden_layers": "Layers", "dropout": "Dropout",
        "learning_rate": "LR", "weight_decay": "WD",
        "batch_size": "BS", "activation": "Act",
        "physics_weight": "Phys-W", "physics_warmup_fraction": "Phys-WF",
        "phi_lr_ratio": "phi-LR", "alpha_reg_weight": "alpha-Reg",
    }

    for ax, mtype in zip(axes, arch_order):
        recs = sorted(groups[mtype], key=lambda r: rmse_scalar(r, "test"))
        recs = recs[:top_k]
        hp_keys_all = _ARCH_HP_KEYS.get(mtype, _GRID_HP_KEYS_FNN)

        varying_hp_keys: list[str] = []
        for k in hp_keys_all:
            vals = set(hp_val_str(rec.get("hyperparams", {}).get(k, "-")) for rec in recs)
            if len(vals) > 1:
                varying_hp_keys.append(k)

        col_headers = (
            ["Rank", "Test RMSE", "Val RMSE", "Test R2", "Test MAE", "Epochs", "ES"]
            + [_HP_HEADER.get(k, k) for k in varying_hp_keys]
        )
        table_data = []
        for rank, rec in enumerate(recs, 1):
            hp = rec.get("hyperparams", {})
            row = [
                str(rank),
                f"{rmse_scalar(rec, 'test'):.4f}",
                f"{rmse_scalar(rec, 'val'):.4f}",
                f"{split_scalar(rec, 'test', 'r2_overall'):.4f}",
                f"{split_scalar(rec, 'test', 'mae_mean'):.4f}",
                str(rec.get("epochs_trained", "?")),
                "Y" if rec.get("stopped_early") else "N",
            ]
            for k in varying_hp_keys:
                v = hp.get(k, "-")
                if k == "learning_rate" and isinstance(v, float):
                    row.append(f"{v:.1e}")
                else:
                    row.append(hp_val_str(v))
            table_data.append(row)

        ax.axis("off")
        tbl = ax.table(cellText=table_data, colLabels=col_headers,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.60)

        for col_idx in range(len(col_headers)):
            cell = tbl[(0, col_idx)]
            cell.set_facecolor("#2c4770")
            cell.set_text_props(color="white", fontweight="bold")

        for row_idx in range(1, len(recs) + 1):
            bg = "#f4f7fb" if row_idx % 2 == 0 else "white"
            for col_idx in range(len(col_headers)):
                tbl[(row_idx, col_idx)].set_facecolor(bg)

        if recs:
            tbl[(1, 1)].set_text_props(fontweight="bold")

        ax.set_title(
            f"{arch_short_label(mtype)}  -  Top {min(top_k, len(recs))} of"
            f" {len(groups[mtype])} runs  (ranked by Test RMSE)",
            fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout(pad=1.0)
    save_fig(fig, output_dir / "fig10_topk_leaderboard.png")
