from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronic_disease_risk.artifacts.exporter import export_web_artifact


if __name__ == "__main__":
    export_web_artifact(
        "demo",
        {"auc": 0.5, "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5},
        ["age", "glm7_score"],
        REPO_ROOT / "artifacts" / "demo.json",
    )
