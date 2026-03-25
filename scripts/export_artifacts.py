from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronic_disease_risk.artifacts.exporter import export_all_training_artifacts
from chronic_disease_risk.config import load_yaml_config


if __name__ == "__main__":
    modeling_config = load_yaml_config(REPO_ROOT / "configs" / "modeling.yaml")
    outputs = export_all_training_artifacts(
        task_names=modeling_config["tasks"],
        reports_dir=REPO_ROOT / "reports" / "tables",
        feature_names=modeling_config["feature_columns"],
        destination_dir=REPO_ROOT / "artifacts",
    )
    print(outputs)
