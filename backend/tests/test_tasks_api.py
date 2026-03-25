from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_tasks_endpoint_returns_task_list() -> None:
    client = TestClient(create_app())

    resp = client.get('/api/tasks')
    assert resp.status_code == 200

    data = resp.json()
    assert 'tasks' in data
    assert isinstance(data['tasks'], list)
    assert len(data['tasks']) >= 1

    # spot-check required keys
    t0 = data['tasks'][0]
    assert 'task' in t0
    assert 'best_model_name' in t0
    assert 'auc' in t0
    assert 'accuracy' in t0
    assert 'precision' in t0
    assert 'recall' in t0
    assert 'f1' in t0