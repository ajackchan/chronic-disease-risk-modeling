from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_candidate_models(random_state: int, *, enable_feature_engineering: bool = False) -> dict[str, object]:
    """Return candidate estimators.

    By default, this returns plain estimators that will be wrapped by the common
    feature pipeline (imputer -> model).

    When `enable_feature_engineering=True`, it adds a polynomial-interaction
    baseline to explore non-linear transforms/interaction terms as required by
    the proposal.
    """

    candidates: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        ),
        "svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
    }

    if enable_feature_engineering:
        candidates["logistic_poly2"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )

    return candidates