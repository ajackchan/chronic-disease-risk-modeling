from pathlib import Path

from chronic_disease_risk.artifacts.exporter import export_web_artifact


def test_pipeline_smoke_writes_metrics_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"

    export_web_artifact("diabetes", {"auc": 0.81}, ["age", "glm7_score"], destination)

    assert destination.exists()
