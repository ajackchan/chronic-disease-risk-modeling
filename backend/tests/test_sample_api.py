from backend.tests.conftest import create_test_client


def test_samples_endpoint_returns_demo_cases() -> None:
    client = create_test_client()

    response = client.get('/api/samples')

    assert response.status_code == 200
    payload = response.json()
    assert len(payload['samples']) >= 3
    assert 'ridageyr' in payload['samples'][0]['inputs']
