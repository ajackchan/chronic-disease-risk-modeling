from backend.tests.conftest import create_test_client


def test_overview_endpoint_returns_five_task_cards() -> None:
    client = create_test_client()

    response = client.get('/api/overview')

    assert response.status_code == 200
    payload = response.json()
    assert len(payload['tasks']) == 5
    assert payload['tasks'][0]['task']
    assert 'best_model_name' in payload['tasks'][0]
