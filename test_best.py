import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from Neural_Networks.analyzer.io.scan import scan_trained_models, group_by_model_type
from Neural_Networks.analyzer.compute.enrich import enrich_records
from Neural_Networks.analyzer.io.records import rmse_scalar

run_dir = "Neural_Networks/Trained_Models/Grid_Searches/Trained_Models_Grid/run_train22_q0_qd91_qdd21_tau51_rnea15"
records = scan_trained_models(run_dir)
enrich_records(records, mode="cached")
groups = group_by_model_type(records)

for mtype, recs in groups.items():
    best_val = min(recs, key=lambda r: rmse_scalar(r, "val"))
    best_test = min(recs, key=lambda r: rmse_scalar(r, "test"))
    print(f"{mtype}:")
    print(f"  Best by Val: Val RMSE={rmse_scalar(best_val, 'val'):.4f}, Test RMSE={rmse_scalar(best_val, 'test'):.4f}")
    print(f"  Best by Test: Val RMSE={rmse_scalar(best_test, 'val'):.4f}, Test RMSE={rmse_scalar(best_test, 'test'):.4f}")
