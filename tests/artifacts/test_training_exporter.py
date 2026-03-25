import json
from pathlib import Path

from chronic_disease_risk.artifacts.exporter import export_training_artifact


def test_export_training_artifact_reads_real_metric_table(tmp_path: Path) -> None:
    summary_path = tmp_path / "candidate_diabetes_summary.csv"
    summary_path.write_text(
        "model_name,auc,accuracy,precision,recall,f1\nlogistic_regression,0.85,0.75,0.44,0.78,0.56\n",
        encoding="utf-8",
    )
    destination = tmp_path / "diabetes_artifact.json"

    export_training_artifact(
        task_name="diabetes",
        metrics_table_path=summary_path,
        feature_names=["ridageyr", "aip", "tyg", "tyg_bmi", "glm7_score"],
        destination=destination,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["task"] == "diabetes"
    assert payload["rows"][0]["model_name"] == "logistic_regression"
    assert payload["features"] == ["ridageyr", "aip", "tyg", "tyg_bmi", "glm7_score"]
