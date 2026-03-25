from chronic_disease_risk.modeling.candidates import build_candidate_models


def test_build_candidate_models_includes_expected_estimators() -> None:
    models = build_candidate_models(random_state=42)

    assert {"logistic_regression", "random_forest", "xgboost", "svm"}.issubset(models)
