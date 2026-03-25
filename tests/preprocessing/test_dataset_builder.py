import math

import pandas as pd
import pytest

from chronic_disease_risk.preprocessing.dataset_builder import build_processed_dataset


def test_build_processed_dataset_adds_features_and_labels() -> None:
    df = pd.DataFrame(
        {
            "seqn": [1],
            "cycle": ["2017-2018"],
            "ridstatr": [2],
            "ridageyr": [45],
            "lbxglu": [100.0],
            "lbxtr": [150.0],
            "lbdhdd": [50.0],
            "lbxin": [12.0],
            "bmxbmi": [24.0],
            "lbdldl": [110.0],
            "mcq160b": [2],
            "mcq160c": [1],
            "mcq160d": [2],
            "mcq160e": [2],
            "mcq160f": [2],
            "mcq160l": [2],
            "mcq220": [2],
            "diq010": [2],
        }
    )

    processed = build_processed_dataset(df)

    assert processed.loc[0, "aip"] == pytest.approx(math.log10(150.0 / 50.0))
    assert processed.loc[0, "tyg_bmi"] > 0
    assert processed.loc[0, "cardiovascular"] == 1
    assert processed.loc[0, "diabetes"] == 0
    assert processed.loc[0, "liver"] == 0
    assert processed.loc[0, "cancer"] == 0
    assert processed.loc[0, "multimorbidity_count"] == 1
    assert "glm7_score" in processed.columns
