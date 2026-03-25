from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def export_web_artifact(task_name: str, metrics: dict[str, float], feature_names: list[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task_name,
        "metrics": metrics,
        "features": feature_names,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_training_artifact(task_name: str, metrics_table_path: Path, feature_names: list[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(metrics_table_path).to_dict(orient="records")
    payload = {
        "task": task_name,
        "features": feature_names,
        "rows": rows,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_all_training_artifacts(
    task_names: list[str],
    reports_dir: Path,
    feature_names: list[str],
    destination_dir: Path,
) -> dict[str, Path]:
    outputs = {}
    for task_name in task_names:
        destination = destination_dir / f"{task_name}_candidate_summary.json"
        export_training_artifact(
            task_name=task_name,
            metrics_table_path=reports_dir / f"candidate_{task_name}_summary.csv",
            feature_names=feature_names,
            destination=destination,
        )
        outputs[task_name] = destination
    return outputs
