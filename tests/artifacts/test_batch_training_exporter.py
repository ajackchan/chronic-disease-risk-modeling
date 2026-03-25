import json
from pathlib import Path

from chronic_disease_risk.artifacts.exporter import export_all_training_artifacts


def test_export_all_training_artifacts_writes_one_json_per_task(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for task in ["cardiovascular", "diabetes"]:
        (reports_dir / f"candidate_{task}_summary.csv").write_text(
            "model_name,auc,accuracy,precision,recall,f1\nlogistic_regression,0.85,0.75,0.44,0.78,0.56\n",
            encoding="utf-8",
        )

    outputs = export_all_training_artifacts(
        task_names=["cardiovascular", "diabetes"],
        reports_dir=reports_dir,
        feature_names=["ridageyr", "aip", "tyg", "tyg_bmi", "glm7_score"],
        destination_dir=tmp_path / "artifacts",
    )

    assert set(outputs) == {"cardiovascular", "diabetes"}
    payload = json.loads(outputs["diabetes"].read_text(encoding="utf-8"))
    assert payload["task"] == "diabetes"
    assert payload["rows"][0]["model_name"] == "logistic_regression"
