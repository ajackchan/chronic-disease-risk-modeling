import numpy as np

from chronic_disease_risk.evaluation.metrics import compute_binary_metrics


def test_compute_binary_metrics_returns_expected_keys() -> None:
    metrics = compute_binary_metrics(
        y_true=np.array([0, 0, 1, 1]),
        y_prob=np.array([0.1, 0.3, 0.7, 0.9]),
        threshold=0.5,
    )

    assert set(metrics) == {"auc", "accuracy", "precision", "recall", "f1"}
