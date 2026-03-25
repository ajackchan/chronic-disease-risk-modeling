from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_logistic_baseline(X, y) -> Pipeline:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    return pipeline.fit(X, y)
