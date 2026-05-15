
import yaml
from pathlib import Path

registry_path = "Neural_Networks/Trained_Models_Grid/run_train22_q0_qd91_qdd21_tau51_rnea15/models_registry.yaml"

with open(registry_path, "r") as f:
    data = yaml.safe_load(f)

best_models = {}

for model in data["models"]:
    m_type = model["model_type"]
    test_rmse = model.get("test_metrics", {}).get("rmse_mean", float("inf"))
    
    if m_type not in best_models or test_rmse < best_models[m_type]["test_rmse"]:
        best_models[m_type] = {
            "test_rmse": test_rmse,
            "hp": model["hyperparams"]
        }

for m_type, info in best_models.items():
    print(f"Architecture: {m_type}")
    print(f"  Best Test RMSE: {info['test_rmse']}")
    hp = info["hp"]
    print(f"  Data Fraction (f): {hp.get('data_train_fraction')}")
    if "physics_weight" in hp:
        print(f"  Physics Weight (w): {hp.get('physics_weight')}")
    if "alpha_reg_weight" in hp:
        print(f"  Alpha Reg Weight (w): {hp.get('alpha_reg_weight')}")
    print("-" * 20)
