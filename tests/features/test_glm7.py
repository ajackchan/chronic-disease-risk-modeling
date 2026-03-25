from chronic_disease_risk.features.glm7 import GLM7Builder


def test_glm7_builder_returns_raw_and_score_columns() -> None:
    builder = GLM7Builder(weights={"age": 0.1, "insulin": 0.2}, intercept=-1.0)
    row = {"age": 50.0, "insulin": 10.0}

    result = builder.transform_row(row)

    assert "glm7_score" in result
    assert result["glm7_score"] > 0
