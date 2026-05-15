import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from Neural_Networks.analyzer.io.scan import scan_trained_models, group_by_model_type
from Neural_Networks.analyzer.compute.enrich import enrich_records
from Neural_Networks.analyzer.io.records import best_per_type, split_joints

run_dir = "Neural_Networks/Trained_Models/Grid_Searches/Trained_Models_Grid/run_train22_q0_qd91_qdd21_tau51_rnea15"
records = scan_trained_models(run_dir)
enrich_records(records, mode="cached")
groups = group_by_model_type(records)
all_recs = best_per_type(groups)
for r in all_recs:
    print(r.get("model_type"))
    print("Test:", split_joints(r, "test", "rmse"))
    print("Train:", split_joints(r, "train", "rmse"))
