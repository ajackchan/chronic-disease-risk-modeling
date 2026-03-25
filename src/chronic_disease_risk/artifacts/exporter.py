from __future__ import annotations

import json
from pathlib import Path


def export_web_artifact(task_name: str, metrics: dict[str, float], feature_names: list[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task_name,
        "metrics": metrics,
        "features": feature_names,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
