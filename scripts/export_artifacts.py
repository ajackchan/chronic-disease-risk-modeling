from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronic_disease_risk.artifacts.exporter import export_training_artifact


if __name__ == "__main__":
    export_training_artifact(
        task_name="diabetes",
        metrics_table_path=REPO_ROOT / "reports" / "tables" / "candidate_diabetes_summary.csv",
        feature_names=["ridageyr", "aip", "tyg", "tyg_bmi", "glm7_score"],
        destination=REPO_ROOT / "artifacts" / "diabetes_candidate_summary.json",
    )
