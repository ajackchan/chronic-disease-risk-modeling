from backend.tests.conftest import create_test_client


def test_predict_endpoint_returns_five_task_probabilities() -> None:
    client = create_test_client()

    response = client.post(
        '/api/predict',
        json={
            'ridageyr': 56,
            'bmxbmi': 24.5,
            'lbxglu': 105.0,
            'lbxtr': 160.0,
            'lbdhdd': 45.0,
            'lbdldl': 118.0,
            'lbxin': 9.5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload['predictions']) == {
        'cardiovascular',
        'diabetes',
        'liver',
        'cancer',
        'multimorbidity_label',
    }
