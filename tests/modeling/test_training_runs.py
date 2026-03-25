from pathlib import Path

import pandas as pd

from chronic_disease_risk.modeling.training_runs import run_baseline_training


def test_run_baseline_training_writes_metrics_model_and_plots(tmp_path: Path) -> None:
    dataset_path = tmp_path / "processed.csv"
    df = pd.DataFrame(
        {
            "cycle": ["2017-2018"] * 6,
            "ridageyr": [40, 45, 50, 55, 60, 65],
            "aip": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "tyg": [7.1, 7.2, 7.3, 7.4, 7.5, 7.6],
            "tyg_bmi": [170, 175, 180, 185, 190, 195],
            "glm7_score": [0.1, 0.2, 0.3, 0.6, 0.7, 0.8],
            "diabetes": [0, 0, 0, 1, 1, 1],
        }
    )
    df.to_csv(dataset_path, index=False)

    output = run_baseline_training(
        dataset_path=dataset_path,
        target_column="diabetes",
        feature_columns=["ridageyr", "aip", "tyg", "tyg_bmi", "glm7_score"],
        output_dir=tmp_path,
    )

    assert output["metrics_path"].exists()
    assert output["model_path"].exists()
    assert output["roc_plot_path"].exists()
    assert output["confusion_matrix_path"].exists()
    metrics = pd.read_csv(output["metrics_path"])
    assert set(metrics.columns) == {"auc", "accuracy", "precision", "recall", "f1"}
