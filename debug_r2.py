import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from Neural_Networks.analyzer.io.scan import scan_trained_models, group_by_model_type
from Neural_Networks.analyzer.compute.enrich import enrich_records
from Neural_Networks.analyzer.io.records import best_per_type, split_scalar, rmse_scalar

run_dir = "Neural_Networks/Trained_Models/Grid_Searches/Trained_Models_Grid/run_train22_q0_qd91_qdd21_tau51_rnea15"
records = scan_trained_models(run_dir)
enrich_records(records, mode="cached")
groups = group_by_model_type(records)
all_recs = best_per_type(groups)
for r in all_recs:
    mtype = r.get("model_type")
    tr_r2 = split_scalar(r, "train", "r2_overall")
    te_r2 = split_scalar(r, "test", "r2_overall")
    tr_rmse = rmse_scalar(r, "train")
    te_rmse = rmse_scalar(r, "test")
    print(f"{mtype}:")
    print(f"  Train R2: {tr_r2:.4f}, Test R2: {te_r2:.4f}")
    print(f"  Train RMSE: {tr_rmse:.4f}, Test RMSE: {te_rmse:.4f}")
