from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_feature_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
