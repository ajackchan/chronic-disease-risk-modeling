import pandas as pd

from chronic_disease_risk.preprocessing.cohort import apply_inclusion_rules


def test_apply_inclusion_rules_keeps_adults_with_required_labs() -> None:
    df = pd.DataFrame(
        {
            "ridageyr": [19, 45],
            "lbxglu": [100.0, 90.0],
            "lbxtr": [150.0, 120.0],
            "lbdhdd": [50.0, 55.0],
            "lbdldl": [110.0, 120.0],
            "lbxins": [10.0, 11.0],
            "bmxbmi": [22.0, 24.0],
        }
    )

    filtered = apply_inclusion_rules(df)

    assert len(filtered) == 1
    assert filtered.iloc[0]["ridageyr"] == 45