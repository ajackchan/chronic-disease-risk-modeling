from backend.tests.conftest import create_test_client


def test_explain_endpoint_works_with_stub(monkeypatch) -> None:
    # Patch at the router module level so we do not depend on SHAP runtime.
    import backend.app.api.routes_explain as routes

    def _stub(task: str, payload, top_k: int = 10) -> dict:
        return {
            "task": task,
            "probability": 0.5,
            "base_value": 0.0,
            "contributions": [
                {"feature": "ridageyr", "value": float(payload.ridageyr), "shap_value": 0.1},
            ],
        }

    monkeypatch.setattr(routes, "explain_single_task", _stub)

    client = create_test_client()

    resp = client.post(
        "/api/explain",
        json={
            "task": "diabetes",
            "top_k": 5,
            "input": {
                "ridageyr": 56,
                "bmxbmi": 24.8,
                "lbxglu": 105.0,
                "lbxtr": 160.0,
                "lbdhdd": 45.0,
                "lbdldl": 118.0,
                "lbxin": 9.5,
            },
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["task"] == "diabetes"
    assert "probability" in data
    assert isinstance(data["contributions"], list)
    assert data["contributions"][0]["feature"] == "ridageyr"