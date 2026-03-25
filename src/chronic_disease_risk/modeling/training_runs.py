from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from chronic_disease_risk.evaluation.metrics import compute_binary_metrics
from chronic_disease_risk.evaluation.reporting import write_metrics_table
from chronic_disease_risk.modeling.baseline import train_logistic_baseline
from chronic_disease_risk.modeling.candidates import build_candidate_models
from chronic_disease_risk.modeling.pipeline import build_feature_pipeline
from chronic_disease_risk.modeling.search import select_best_model


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
    metrics = compute_binary_metrics(y_test.to_numpy(), y_prob)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"baseline_{target_column}_metrics.csv"
    model_path = output_dir / f"baseline_{target_column}.joblib"
    write_metrics_table(metrics, metrics_path)
    joblib.dump(model, model_path)

    return {"metrics_path": metrics_path, "model_path": model_path}


def run_candidate_training(
    dataset_path: Path,
    target_column: str,
    feature_columns: list[str],
    output_dir: Path,
    random_state: int = 42,
) -> dict[str, Path | str]:
    X_train, X_test, y_train, y_test = _train_test_data(dataset_path, target_column, feature_columns, random_state)
    candidates = build_candidate_models(random_state=random_state)

    rows = []
    trained_models = {}
    for model_name, estimator in candidates.items():
        pipeline = build_feature_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_binary_metrics(y_test.to_numpy(), y_prob)
        rows.append({"model_name": model_name, **metrics})
        trained_models[model_name] = pipeline

    summary = pd.DataFrame(rows)
    scores = {row["model_name"]: {k: row[k] for k in ["auc", "accuracy", "precision", "recall", "f1"]} for row in rows}
    best_model_name, _ = select_best_model(scores, metric_name="auc")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"candidate_{target_column}_summary.csv"
    model_path = output_dir / f"candidate_best_{target_column}.joblib"
    summary.to_csv(summary_path, index=False)
    joblib.dump(trained_models[best_model_name], model_path)

    return {
        "summary_path": summary_path,
        "best_model_name": best_model_name,
        "model_path": model_path,
    }
