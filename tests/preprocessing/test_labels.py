import pandas as pd

from chronic_disease_risk.preprocessing.labels import add_multimorbidity_label


def test_add_multimorbidity_label_counts_positive_outcomes() -> None:
    df = pd.DataFrame(
        {
            "cardiovascular": [1, 0],
            "diabetes": [1, 1],
            "liver": [0, 0],
        }
    )

    labeled = add_multimorbidity_label(df, ["cardiovascular", "diabetes", "liver"])

    assert labeled["multimorbidity_count"].tolist() == [2, 1]
    assert labeled["multimorbidity_label"].tolist() == [1, 0]
