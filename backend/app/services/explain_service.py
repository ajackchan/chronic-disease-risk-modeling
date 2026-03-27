from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.config import CONFIGS_DIR, REPORTS_DIR, REPO_ROOT
from backend.app.schemas.predict import PredictRequest
from chronic_disease_risk.config import load_yaml_config

try:  # optional dependency in some environments
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None


def _risk_label(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


def _build_feature_row(payload: PredictRequest) -> pd.DataFrame:
    # Reuse backend inference feature building.
    from backend.app.services.prediction_service import _build_feature_row as _build

    return _build(payload)


@lru_cache(maxsize=1)
def _load_feature_columns() -> list[str]:
    modeling_config = load_yaml_config(CONFIGS_DIR / "modeling.yaml")
    return list(modeling_config.get("feature_columns", []))


@lru_cache(maxsize=1)
def _load_background_df() -> pd.DataFrame:
    """Small background sample for SHAP."""

    dataset_path = Path(REPO_ROOT) / "data" / "processed" / "nhanes_model_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/nhanes_model_dataset.csv. "
            "SHAP explanation requires a background dataset sample."
        )

    df = pd.read_csv(dataset_path)
    cols = _load_feature_columns()
    bg = df[cols]

    if len(bg) > 200:
        bg = bg.sample(n=200, random_state=42)

    return bg


@lru_cache(maxsize=32)
def _load_task_pipeline(task_name: str):
    model_path = Path(REPORTS_DIR) / f"candidate_best_{task_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path.as_posix()}")
    return joblib.load(model_path)


def _resolve_pipeline_for_shap(pipeline, X: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    named_steps = getattr(pipeline, "named_steps", None)
    if not named_steps:
        return pipeline, X

    imputer = named_steps.get("imputer")
    estimator = named_steps.get("model")
    if imputer is None or estimator is None:
        return pipeline, X

    X_imp = imputer.transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=list(X.columns))
    return estimator, X_imp_df


def explain_single_task(task_name: str, payload: PredictRequest, *, top_k: int = 10) -> dict:
    if shap is None:
        raise RuntimeError("shap is not installed; cannot compute explanations")

    feature_columns = _load_feature_columns()

    row_full = _build_feature_row(payload)
    row = row_full[feature_columns]

    pipeline = _load_task_pipeline(task_name)
    probability = float(pipeline.predict_proba(row)[0, 1])

    background = _load_background_df()
    estimator, bg_imp = _resolve_pipeline_for_shap(pipeline, background)
    _, row_imp = _resolve_pipeline_for_shap(pipeline, row)

    explainer = shap.Explainer(estimator, bg_imp)
    shap_values = explainer(row_imp)

    # Binary classifiers may produce (n, features, outputs). Keep positive class.
    try:
        if getattr(shap_values, "values", None) is not None and getattr(shap_values.values, "ndim", 0) == 3:
            shap_values = shap_values[..., 1]
    except Exception:
        pass

    values = np.asarray(getattr(shap_values, "values", None))
    if values.ndim != 2 or values.shape[0] != 1:
        raise RuntimeError("Unexpected SHAP output shape")

    base_value = None
    try:
        base_values = getattr(shap_values, "base_values", None)
        if base_values is not None:
            base_arr = np.asarray(base_values)
            if base_arr.ndim == 2 and base_arr.shape[0] == 1:
                base_value = float(base_arr[0, -1])
            elif base_arr.ndim == 1 and base_arr.shape[0] == 1:
                base_value = float(base_arr[0])
    except Exception:
        base_value = None

    contribs = []
    for i, col in enumerate(feature_columns):
        contribs.append(
            {
                "feature": col,
                "value": float(row[col].iloc[0]),
                "shap_value": float(values[0, i]),
            }
        )

    contribs.sort(key=lambda r: abs(float(r["shap_value"])), reverse=True)
    contribs = contribs[: max(1, int(top_k))]

    return {
        "task": task_name,
        "probability": probability,
        "base_value": base_value,
        "contributions": contribs,
        "risk_label": _risk_label(probability),
    }