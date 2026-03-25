from backend.tests.conftest import create_test_client


def test_task_artifacts_endpoint_returns_chart_urls() -> None:
    client = create_test_client()

    response = client.get('/api/tasks/diabetes/artifacts')

    assert response.status_code == 200
    payload = response.json()
    assert payload['task'] == 'diabetes'
    assert payload['roc_plot_url'].endswith('baseline_diabetes_roc.png')
    assert payload['comparison_plot_url'].endswith('candidate_diabetes_comparison.png')
