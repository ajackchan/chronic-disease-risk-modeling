from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronic_disease_risk.config import load_yaml_config
from chronic_disease_risk.modeling.training_runs import run_all_candidate_trainings


if __name__ == "__main__":
    modeling_config = load_yaml_config(REPO_ROOT / "configs" / "modeling.yaml")

    # Default: randomized search. Add --grid to switch to GridSearchCV.
    tuning_mode = "grid" if "--grid" in sys.argv else "random"

    output = run_all_candidate_trainings(
        dataset_path=REPO_ROOT / "data" / "processed" / "nhanes_model_dataset.csv",
        task_names=modeling_config["tasks"],
        feature_columns=modeling_config["feature_columns"],
        output_dir=REPO_ROOT / "reports" / "tables",
        random_state=modeling_config.get("random_state", 42),
        enable_tuning=True,
        tuning_mode=tuning_mode,
    )
    print(output)