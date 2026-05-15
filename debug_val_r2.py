import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from Neural_Networks.analyzer.io.scan import scan_trained_models, group_by_model_type
from Neural_Networks.analyzer.io.records import best_per_type, split_scalar

run_dir = "Neural_Networks/Trained_Models/Grid_Searches/Trained_Models_Grid/run_train22_q0_qd91_qdd21_tau51_rnea15"
records = scan_trained_models(run_dir)
groups = group_by_model_type(records)
all_recs = best_per_type(groups)
for r in all_recs:
    mtype = r.get("model_type")
    te_r2 = split_scalar(r, "test", "r2_overall")
    va_r2 = split_scalar(r, "val", "r2_overall")
    print(f"{mtype}: Test R2={te_r2:.4f}, Val R2={va_r2:.4f}")
