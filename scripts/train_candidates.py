from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronic_disease_risk.modeling.training_runs import run_candidate_training


if __name__ == "__main__":
    output = run_candidate_training(
        dataset_path=REPO_ROOT / "data" / "processed" / "nhanes_model_dataset.csv",
        target_column="diabetes",
        feature_columns=["ridageyr", "aip", "tyg", "tyg_bmi", "glm7_score"],
        output_dir=REPO_ROOT / "reports" / "tables",
        random_state=42,
    )
    print(output)
