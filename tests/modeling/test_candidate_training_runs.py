from pathlib import Path

import pandas as pd

from chronic_disease_risk.modeling.training_runs import run_candidate_training


def test_run_candidate_training_selects_best_model_and_writes_summary(tmp_path: Path) -> None:
    dataset_path = tmp_path / "processed.csv"
    df = pd.DataFrame(
        {
            "cycle": ["2017-2018"] * 8,
            "ridageyr": [40, 42, 45, 48, 52, 56, 60, 64],
            "aip": [0.1, 0.12, 0.2, 0.25, 0.4, 0.42, 0.55, 0.6],
            "tyg": [7.0, 7.1, 7.2, 7.3, 7.5, 7.6, 7.8, 7.9],
            "tyg_bmi": [160, 162, 170, 175, 185, 190, 205, 210],
            "glm7_score": [0.1, 0.15, 0.22, 0.3, 0.6, 0.65, 0.75, 0.8],
            "diabetes": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    df.to_csv(dataset_path, index=False)

    output = run_candidate_training(
        dataset_path=dataset_path,
        target_column="diabetes",
        feature_columns=["ridageyr", "aip", "tyg", "tyg_bmi", "glm7_score"],
        output_dir=tmp_path,
        random_state=42,
    )

    assert output["summary_path"].exists()
    assert output["best_model_name"]
    summary = pd.read_csv(output["summary_path"])
    assert "model_name" in summary.columns
    assert "auc" in summary.columns
