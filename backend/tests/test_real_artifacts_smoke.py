from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.services.overview_service import get_task_names, load_task_overview


def _assert_file_exists(path: Path) -> None:
    assert path.exists(), f"Missing file: {path.as_posix()}"
    assert path.is_file(), f"Not a file: {path.as_posix()}"


def test_reports_and_models_exist_for_all_tasks() -> None:
    """Guardrail for defense: ensure the demo has the real artifacts it expects."""

    repo_root = Path(__file__).resolve().parents[2]
    reports = repo_root / "reports" / "tables"

    tasks = get_task_names()
    assert tasks, "No tasks found in configs/modeling.yaml"

    for task in tasks:
        # Overview CSV must exist and be parseable.
        overview = load_task_overview(task, reports_dir=reports)
        assert 0.0 <= overview["auc"] <= 1.0
        assert overview["auc"] > 0.5, f"AUC too low for task={task}: {overview['auc']}"

        _assert_file_exists(reports / f"candidate_{task}_summary.csv")
        _assert_file_exists(reports / f"candidate_{task}_comparison.png")
        _assert_file_exists(reports / f"baseline_{task}_roc.png")
        _assert_file_exists(reports / f"baseline_{task}_confusion.png")
        _assert_file_exists(reports / f"candidate_best_{task}.joblib")


def test_demo_api_predict_returns_all_tasks() -> None:
    client = TestClient(create_app())

    payload = {
        "ridageyr": 56,
        "bmxbmi": 24.8,
        "lbxglu": 105.0,
        "lbxtr": 160.0,
        "lbdhdd": 45.0,
        "lbdldl": 118.0,
        "lbxin": 9.5,
    }

    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "predictions" in data

    preds = data["predictions"]
    tasks = get_task_names()

    for task in tasks:
        assert task in preds, f"Missing prediction for task={task}"
        p = preds[task]["probability"]
        assert 0.0 <= p <= 1.0
        assert preds[task]["risk_label"] in {"low", "medium", "high"}