from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from backend.app.config import CONFIGS_DIR, REPORTS_DIR
from backend.app.schemas.predict import PredictRequest
from chronic_disease_risk.config import load_yaml_config
from chronic_disease_risk.features.formulas import compute_aip, compute_tyg, compute_tyg_bmi
from chronic_disease_risk.features.glm7 import GLM7Builder


def _build_feature_row(payload: PredictRequest) -> pd.DataFrame:
    glm7_config = load_yaml_config(CONFIGS_DIR / 'glm7.yaml')
    aip = compute_aip(payload.lbxtr, payload.lbdhdd)
    tyg = compute_tyg(payload.lbxtr, payload.lbxglu)
    tyg_bmi = compute_tyg_bmi(tyg, payload.bmxbmi)
    glm7_builder = GLM7Builder(weights=glm7_config.get('weights', {}), intercept=glm7_config.get('intercept', 0.0))
    glm7_score = glm7_builder.transform_row(
        {
            'age': payload.ridageyr,
            'insulin': payload.lbxin,
            'triglycerides': payload.lbxtr,
            'ldl_c': payload.lbdldl,
            'aip': aip,
            'tyg': tyg,
            'tyg_bmi': tyg_bmi,
        }
    )['glm7_score']
    return pd.DataFrame(
        [
            {
                'ridageyr': payload.ridageyr,
                'aip': aip,
                'tyg': tyg,
                'tyg_bmi': tyg_bmi,
                'glm7_score': glm7_score,
            }
        ]
    )


def _risk_label(probability: float) -> str:
    if probability >= 0.7:
        return 'high'
    if probability >= 0.4:
        return 'medium'
    return 'low'


def predict_all_tasks(payload: PredictRequest, model_dir: Path = REPORTS_DIR) -> dict[str, dict]:
    modeling_config = load_yaml_config(CONFIGS_DIR / 'modeling.yaml')
    task_names = modeling_config.get('tasks', [])
    feature_row = _build_feature_row(payload)
    predictions = {}

    for task_name in task_names:
        model = joblib.load(model_dir / f'candidate_best_{task_name}.joblib')
        probability = float(model.predict_proba(feature_row)[0, 1])
        predictions[task_name] = {
            'probability': probability,
            'risk_label': _risk_label(probability),
        }

    return predictions
