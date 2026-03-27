from chronic_disease_risk.modeling.candidates import build_candidate_models


def test_build_candidate_models_can_enable_feature_engineering() -> None:
    base = build_candidate_models(random_state=42)
    assert "logistic_poly2" not in base

    fe = build_candidate_models(random_state=42, enable_feature_engineering=True)
    assert "logistic_poly2" in fe