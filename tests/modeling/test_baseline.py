import pandas as pd

from chronic_disease_risk.modeling.baseline import train_logistic_baseline


def test_train_logistic_baseline_returns_sklearn_pipeline() -> None:
    df = pd.DataFrame(
        {
            "age": [40, 50, 60, 70],
            "glm7_score": [0.1, 0.3, 0.5, 0.7],
            "target": [0, 0, 1, 1],
        }
    )

    model = train_logistic_baseline(df[["age", "glm7_score"]], df["target"])

    assert hasattr(model, "predict_proba")
