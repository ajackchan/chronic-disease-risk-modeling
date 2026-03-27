from pathlib import Path

import numpy as np
import pandas as pd

from chronic_disease_risk.evaluation.reporting import (
    save_calibration_curve_plot,
    save_confusion_matrix_plot,
    save_decision_curve_plot,
    save_model_comparison_plot,
    save_roc_curve_plot,
)


def test_save_roc_curve_plot_writes_png(tmp_path: Path) -> None:
    destination = tmp_path / "roc.png"

    save_roc_curve_plot(
        y_true=np.array([0, 0, 1, 1]),
        y_prob=np.array([0.1, 0.3, 0.7, 0.9]),
        destination=destination,
        title="ROC",
    )

    assert destination.exists()
    assert destination.stat().st_size > 0


def test_save_confusion_matrix_plot_writes_png(tmp_path: Path) -> None:
    destination = tmp_path / "confusion.png"

    save_confusion_matrix_plot(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 1]),
        destination=destination,
        title="Confusion",
    )

    assert destination.exists()
    assert destination.stat().st_size > 0


def test_save_model_comparison_plot_writes_png(tmp_path: Path) -> None:
    destination = tmp_path / "comparison.png"
    summary = pd.DataFrame(
        {
            "model_name": ["logistic_regression", "xgboost"],
            "auc": [0.85, 0.81],
        }
    )

    save_model_comparison_plot(summary, destination=destination, metric_name="auc", title="Model Comparison")

    assert destination.exists()
    assert destination.stat().st_size > 0


def test_save_calibration_curve_plot_writes_png(tmp_path: Path) -> None:
    destination = tmp_path / "calibration.png"

    save_calibration_curve_plot(
        y_true=np.array([0, 0, 1, 1]),
        y_prob=np.array([0.1, 0.2, 0.8, 0.9]),
        destination=destination,
        title="Calibration",
    )

    assert destination.exists()
    assert destination.stat().st_size > 0


def test_save_decision_curve_plot_writes_png(tmp_path: Path) -> None:
    destination = tmp_path / "dca.png"

    save_decision_curve_plot(
        y_true=np.array([0, 0, 1, 1]),
        y_prob=np.array([0.1, 0.2, 0.8, 0.9]),
        destination=destination,
        title="DCA",
    )

    assert destination.exists()
    assert destination.stat().st_size > 0