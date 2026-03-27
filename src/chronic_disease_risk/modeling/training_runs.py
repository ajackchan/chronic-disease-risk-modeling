from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from chronic_disease_risk.evaluation.metrics import compute_binary_metrics
from chronic_disease_risk.evaluation.reporting import (
    save_calibration_curve_plot,
    save_confusion_matrix_plot,
    save_decision_curve_plot,
    save_model_comparison_plot,
    save_roc_curve_plot,
    write_metrics_table,
)
from chronic_disease_risk.modeling.baseline import train_logistic_baseline
from chronic_disease_risk.modeling.candidates import build_candidate_models
from chronic_disease_risk.modeling.pipeline import build_feature_pipeline
from chronic_disease_risk.modeling.search import select_best_model
from chronic_disease_risk.modeling.tuning import tune_xgboost_pipeline


def _train_test_data(dataset_path: Path, target_column: str, feature_columns: list[str], random_state: int = 42):
    df = pd.read_csv(dataset_path)
    X = df[feature_columns]
    y = df[target_column]
    return train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)


def run_baseline_training(
    dataset_path: Path,
    target_column: str,
    feature_columns: list[str],
    output_dir: Path,
    random_state: int = 42,
) -> dict[str, Path]:
    X_train, X_test, y_train, y_test = _train_test_data(dataset_path, target_column, feature_columns, random_state)
    model = train_logistic_baseline(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_binary_metrics(y_test.to_numpy(), y_prob)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"baseline_{target_column}_metrics.csv"
    model_path = output_dir / f"baseline_{target_column}.joblib"

    roc_plot_path = output_dir / f"baseline_{target_column}_roc.png"
    confusion_matrix_path = output_dir / f"baseline_{target_column}_confusion.png"

    calibration_plot_path = output_dir / f"baseline_{target_column}_calibration.png"
    dca_plot_path = output_dir / f"baseline_{target_column}_dca.png"

    write_metrics_table(metrics, metrics_path)
    save_roc_curve_plot(y_test.to_numpy(), y_prob, roc_plot_path, title=f"Baseline ROC - {target_column}")
    save_confusion_matrix_plot(y_test.to_numpy(), y_pred, confusion_matrix_path, title=f"Baseline Confusion - {target_column}")
    save_calibration_curve_plot(y_test.to_numpy(), y_prob, calibration_plot_path, title=f"Baseline Calibration - {target_column}")
    save_decision_curve_plot(y_test.to_numpy(), y_prob, dca_plot_path, title=f"Baseline DCA - {target_column}")

    joblib.dump(model, model_path)

    return {
        "metrics_path": metrics_path,
        "model_path": model_path,
        "roc_plot_path": roc_plot_path,
        "confusion_matrix_path": confusion_matrix_path,
        "calibration_plot_path": calibration_plot_path,
        "dca_plot_path": dca_plot_path,
    }


def run_candidate_training(
    dataset_path: Path,
    target_column: str,
    feature_columns: list[str],
    output_dir: Path,
    random_state: int = 42,
    enable_tuning: bool = False,
    tuning_mode: str = "random",
) -> dict[str, Path | str]:
    X_train, X_test, y_train, y_test = _train_test_data(dataset_path, target_column, feature_columns, random_state)
    candidates = build_candidate_models(random_state=random_state)

    rows: list[dict[str, float | str]] = []
    trained_models: dict[str, object] = {}

    for model_name, estimator in candidates.items():
        pipeline = build_feature_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_binary_metrics(y_test.to_numpy(), y_prob)
        rows.append({"model_name": model_name, **metrics})
        trained_models[model_name] = pipeline

    # Proposal requirement: CV + hyperparameter search.
    # Keep this optional so unit tests and quick runs remain fast.
    if enable_tuning and "xgboost" in candidates:
        xgb_pipeline = trained_models.get("xgboost")
        if xgb_pipeline is not None:
            tuned_path = output_dir / f"candidate_{target_column}_xgboost_tuned_params.json"
            tuned_pipeline, _, _ = tune_xgboost_pipeline(
                xgb_pipeline,
                X_train,
                y_train,
                random_state=random_state,
                search_mode=tuning_mode,
                destination=tuned_path,
            )
            y_prob_tuned = tuned_pipeline.predict_proba(X_test)[:, 1]
            tuned_metrics = compute_binary_metrics(y_test.to_numpy(), y_prob_tuned)
            rows.append({"model_name": "xgboost_tuned", **tuned_metrics})
            trained_models["xgboost_tuned"] = tuned_pipeline

    summary = pd.DataFrame(rows)
    scores = {row["model_name"]: {k: float(row[k]) for k in ["auc", "accuracy", "precision", "recall", "f1"]} for row in rows}
    best_model_name, _ = select_best_model(scores, metric_name="auc")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"candidate_{target_column}_summary.csv"
    model_path = output_dir / f"candidate_best_{target_column}.joblib"
    comparison_plot_path = output_dir / f"candidate_{target_column}_comparison.png"

    best_roc_plot_path = output_dir / f"candidate_best_{target_column}_roc.png"
    best_confusion_matrix_path = output_dir / f"candidate_best_{target_column}_confusion.png"
    best_calibration_plot_path = output_dir / f"candidate_best_{target_column}_calibration.png"
    best_dca_plot_path = output_dir / f"candidate_best_{target_column}_dca.png"

    summary.to_csv(summary_path, index=False)
    save_model_comparison_plot(summary, destination=comparison_plot_path, metric_name="auc", title=f"Candidate Comparison - {target_column}")

    best_pipeline = trained_models[best_model_name]
    y_prob_best = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred_best = (y_prob_best >= 0.5).astype(int)

    save_roc_curve_plot(y_test.to_numpy(), y_prob_best, best_roc_plot_path, title=f"Candidate Best ROC - {target_column}")
    save_confusion_matrix_plot(
        y_test.to_numpy(),
        y_pred_best,
        best_confusion_matrix_path,
        title=f"Candidate Best Confusion - {target_column}",
    )
    save_calibration_curve_plot(
        y_test.to_numpy(),
        y_prob_best,
        best_calibration_plot_path,
        title=f"Candidate Best Calibration - {target_column}",
    )
    save_decision_curve_plot(
        y_test.to_numpy(),
        y_prob_best,
        best_dca_plot_path,
        title=f"Candidate Best DCA - {target_column}",
    )

    joblib.dump(best_pipeline, model_path)

    return {
        "summary_path": summary_path,
        "best_model_name": best_model_name,
        "model_path": model_path,
        "comparison_plot_path": comparison_plot_path,
        "best_roc_plot_path": best_roc_plot_path,
        "best_confusion_matrix_path": best_confusion_matrix_path,
        "best_calibration_plot_path": best_calibration_plot_path,
        "best_dca_plot_path": best_dca_plot_path,
    }


def run_all_baseline_trainings(
    dataset_path: Path,
    task_names: list[str],
    feature_columns: list[str],
    output_dir: Path,
    random_state: int = 42,
) -> dict[str, dict[str, Path]]:
    return {
        task_name: run_baseline_training(dataset_path, task_name, feature_columns, output_dir, random_state)
        for task_name in task_names
    }


def run_all_candidate_trainings(
    dataset_path: Path,
    task_names: list[str],
    feature_columns: list[str],
    output_dir: Path,
    random_state: int = 42,
    enable_tuning: bool = False,
    tuning_mode: str = "random",
) -> dict[str, dict[str, Path | str]]:
    return {
        task_name: run_candidate_training(
            dataset_path=dataset_path,
            target_column=task_name,
            feature_columns=feature_columns,
            output_dir=output_dir,
            random_state=random_state,
            enable_tuning=enable_tuning,
            tuning_mode=tuning_mode,
        )
        for task_name in task_names
    }