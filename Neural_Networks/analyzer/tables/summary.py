"""Console summary table — best-of-everything view for grid runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io.records import split_scalar
from ..plots._common import fmt as _fmt


def print_summary_table(groups: dict[str, list[dict[str, Any]]]) -> None:
    rows: list[dict[str, Any]] = []
    for model_type, recs in groups.items():
        for rec in recs:
            training = rec.get("training", {})
            epochs = rec.get("epochs_trained", training.get("epochs_ran", "?"))
            stopped = rec.get("stopped_early", training.get("stopped_early", False))
            device = rec.get("device", rec.get("hardware", {}).get("device", "?"))
            rows.append({
                "model_type": model_type,
                "run_id": rec.get("run_id", "unknown"),
                "epochs": epochs,
                "early": "Y" if stopped else "N",
                "test_rmse":    split_scalar(rec, "test", "rmse_pooled"),
                "val_rmse":     split_scalar(rec, "val",  "rmse_pooled"),
                "test_r2":      split_scalar(rec, "test", "r2_overall"),
                "val_r2":       split_scalar(rec, "val",  "r2_overall"),
                "test_mae":     split_scalar(rec, "test", "mae_mean"),
                "val_mae":      split_scalar(rec, "val",  "mae_mean"),
                "test_pearson": split_scalar(rec, "test", "pearson_r_mean"),
                "val_pearson":  split_scalar(rec, "val",  "pearson_r_mean"),
                "train_rmse_hist": rec.get("_train_rmse_hist", float("nan")),
                "val_rmse_hist":   rec.get("_val_rmse_hist",   float("nan")),
                "best_epoch": rec.get("_best_epoch", -1),
                "device": device,
            })

    rows.sort(key=lambda r: r["test_rmse"] if r["test_rmse"] == r["test_rmse"] else 999.0)

    W = 148
    print("\n" + "=" * W)
    print("  GRID SEARCH - TRAINED MODELS - PERFORMANCE REPORT")
    print("  val/test RMSE, R2, MAE, Pearson read from proper held-out splits  |  RMSE & MAE in N.m")
    print("  + train_rmse_hist and val_rmse_hist are from training_history.csv at best checkpoint")
    print("=" * W)

    MT = 28
    hdr = (
        f"  {'#':<3} {'Model Type':<{MT}}  "
        f"{'Ep':>5}  {'ES':>3}  "
        f"{'Test RMSE':>11}  {'Val RMSE':>11}  "
        f"{'Test R2':>10}  {'Val R2':>10}  "
        f"{'Test MAE':>10}  {'Val MAE':>10}  "
        f"{'Test P':>9}  {'Val P':>9}  "
        f"{'Tr-RMSE+':>10}  {'V-RMSE+':>10}"
    )
    print(hdr)
    print("-" * W)
    for i, row in enumerate(rows, 1):
        print(
            f"  {i:<3} {row['model_type']:<{MT}}  "
            f"{str(row['epochs']):>5}  {row['early']:>3}  "
            f"{_fmt(row['test_rmse']):>11}  {_fmt(row['val_rmse']):>11}  "
            f"{_fmt(row['test_r2'], 4):>10}  {_fmt(row['val_r2'], 4):>10}  "
            f"{_fmt(row['test_mae']):>10}  {_fmt(row['val_mae']):>10}  "
            f"{_fmt(row['test_pearson'], 4):>9}  {_fmt(row['val_pearson'], 4):>9}  "
            f"{_fmt(row['train_rmse_hist']):>10}  {_fmt(row['val_rmse_hist']):>10}"
        )
    print("-" * W)
    print("  ES=Y: early stopped   +: training-history units\n")

    print("=== Best per model type (ranked by test RMSE, N.m) ===")
    for mtype in sorted(groups.keys()):
        best = min(groups[mtype], key=lambda r: split_scalar(r, "test", "rmse_pooled"))
        bp  = split_scalar(best, "test", "rmse_pooled")
        r2  = split_scalar(best, "test", "r2_overall")
        mae = split_scalar(best, "test", "mae_mean")
        print(
            f"  {mtype:<35}  test RMSE={bp:.5f} N.m  "
            f"test R2={r2:.4f}  test MAE={mae:.5f} N.m  ->  {best.get('run_id', '?')}"
        )
    print()
