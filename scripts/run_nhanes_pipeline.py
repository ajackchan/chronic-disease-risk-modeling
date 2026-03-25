from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronic_disease_risk.artifacts.exporter import export_web_artifact
from chronic_disease_risk.config import load_yaml_config, resolve_repo_path
from chronic_disease_risk.evaluation.reporting import write_metrics_table


def ensure_directories(repo_root: Path) -> dict[str, Path]:
    paths = load_yaml_config(resolve_repo_path(repo_root, "configs/paths.yaml"))
    resolved = {name: resolve_repo_path(repo_root, rel_path) for name, rel_path in paths.items()}

    for path in resolved.values():
        path.mkdir(parents=True, exist_ok=True)

    (resolved["reports_dir"] / "figures").mkdir(parents=True, exist_ok=True)
    (resolved["reports_dir"] / "tables").mkdir(parents=True, exist_ok=True)
    return resolved


def run_pipeline(repo_root: Path | None = None) -> None:
    repo_root = repo_root or REPO_ROOT
    paths = ensure_directories(repo_root)
    metrics = {"auc": 0.5, "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}

    write_metrics_table(metrics, paths["reports_dir"] / "tables" / "smoke_metrics.csv")
    export_web_artifact(
        "smoke_task",
        metrics,
        ["age", "glm7_score"],
        paths["artifacts_dir"] / "smoke_task.json",
    )


if __name__ == "__main__":
    run_pipeline()
