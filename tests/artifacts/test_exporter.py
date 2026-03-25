import json
from pathlib import Path

from chronic_disease_risk.artifacts.exporter import export_web_artifact


def test_export_web_artifact_writes_json_payload(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"

    export_web_artifact("diabetes", {"auc": 0.81}, ["age", "glm7_score"], destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["task"] == "diabetes"
    assert payload["metrics"]["auc"] == 0.81
    assert payload["features"] == ["age", "glm7_score"]
