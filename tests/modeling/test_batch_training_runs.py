from pathlib import Path

import pandas as pd

from chronic_disease_risk.modeling.training_runs import run_all_baseline_trainings, run_all_candidate_trainings


def _write_multitask_dataset(dataset_path: Path) -> None:
    df = pd.DataFrame(
        {
            "cycle": ["2017-2018"] * 12,
            "ridageyr": [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62],
            "aip": [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50],
            "tyg": [7.0, 7.05, 7.1, 7.15, 7.2, 7.25, 7.6, 7.65, 7.7, 7.75, 7.8, 7.85],
            "tyg_bmi": [160, 162, 165, 168, 170, 172, 195, 198, 202, 205, 208, 210],
            "glm7_score": [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7],
            "cardiovascular": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
            "diabetes": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            "liver": [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            "cancer": [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
            "multimorbidity_label": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
        }
    )
    df.to_csv(dataset_path, index=False)


def test_run_all_baseline_trainings_writes_one_result_per_task(tmp_path: Path) -> None:
    dataset_path = tmp_path / "processed.csv"
    _write_multitask_dataset(dataset_path)

    outputs = run_all_baseline_trainings(
        dataset_path=dataset_path,
        task_names=["cardiovascular", "diabetes", "liver", "cancer", "multimorbidity_label"],
        feature_columns=["ridageyr", "aip", "tyg", "tyg_bmi", "glm7_score"],
        output_dir=tmp_path / "baseline",
    )

    assert set(outputs) == {"cardiovascular", "diabetes", "liver", "cancer", "multimorbidity_label"}
    assert all(result["metrics_path"].exists() for result in outputs.values())
    assert all(result["model_path"].exists() for result in outputs.values())
    assert all(result["roc_plot_path"].exists() for result in outputs.values())
    assert all(result["confusion_matrix_path"].exists() for result in outputs.values())


def test_run_all_candidate_trainings_writes_one_summary_per_task(tmp_path: Path) -> None:
    dataset_path = tmp_path / "processed.csv"
    _write_multitask_dataset(dataset_path)

    outputs = run_all_candidate_trainings(
        dataset_path=dataset_path,
        task_names=["cardiovascular", "diabetes", "liver", "cancer", "multimorbidity_label"],
        feature_columns=["ridageyr", "aip", "tyg", "tyg_bmi", "glm7_score"],
        output_dir=tmp_path / "candidate",
        random_state=42,
    )

    assert set(outputs) == {"cardiovascular", "diabetes", "liver", "cancer", "multimorbidity_label"}
    assert all(result["summary_path"].exists() for result in outputs.values())
    assert all(result["model_path"].exists() for result in outputs.values())
    assert all(result["comparison_plot_path"].exists() for result in outputs.values())
