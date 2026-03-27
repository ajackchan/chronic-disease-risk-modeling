from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def tune_xgboost_pipeline(
    pipeline,
    X_train,
    y_train,
    random_state: int,
    destination: Path | None = None,
    n_iter: int = 8,
    cv_splits: int = 3,
) -> tuple[object, dict, float]:
    """Tune an sklearn Pipeline that ends with an XGBoost classifier.

    This keeps the search light enough for a thesis project while still meeting
    the proposal requirement of 'CV + grid/random search'.

    The pipeline is expected to have a final step named 'model'.
    """

    param_distributions = {
        'model__n_estimators': [120, 180, 240, 300],
        'model__max_depth': [3, 4, 5],
        'model__learning_rate': [0.03, 0.05, 0.08, 0.1],
        'model__subsample': [0.75, 0.85, 0.95],
        'model__colsample_bytree': [0.75, 0.85, 0.95],
        'model__min_child_weight': [1, 3, 5],
        'model__reg_lambda': [0.8, 1.0, 1.2],
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_estimator = search.best_estimator_
    best_params = dict(search.best_params_)
    best_score = float(search.best_score_)

    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'best_score_cv_auc': best_score,
            'best_params': best_params,
        }
        destination.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    return best_estimator, best_params, best_score