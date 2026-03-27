from pathlib import Path

import pandas as pd
import pytest

from chronic_disease_risk.modeling.time_split import choose_time_split_cycles, split_train_test_by_time


def test_choose_time_split_cycles_uses_latest_cycles_for_test() -> None:
    df = pd.DataFrame(
        {
            "cycle": ["2013-2014"] * 3 + ["2015-2016"] * 3 + ["2017-2018"] * 4,
            "ridageyr": list(range(10)),
            "aip": [0.1] * 10,
            "diabetes": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
        }
    )

    plan = choose_time_split_cycles(df, test_size=0.3)
    # Latest cycle must be in test.
    assert "2017-2018" in plan.test_cycles
    # Oldest cycle must be in train.
    assert "2013-2014" in plan.train_cycles


def test_split_train_test_by_time_returns_disjoint_cycles() -> None:
    df = pd.DataFrame(
        {
            "cycle": ["2013-2014"] * 5 + ["2015-2016"] * 5,
            "ridageyr": list(range(10)),
            "aip": [0.1] * 10,
            "diabetes": [0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
        }
    )

    X_train, X_test, y_train, y_test, plan = split_train_test_by_time(
        df,
        feature_columns=["ridageyr", "aip"],
        target_column="diabetes",
        test_size=0.4,
    )

    assert set(X_train.columns) == {"ridageyr", "aip"}
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    train_cycles = set(df.loc[X_train.index, "cycle"].astype(str).unique().tolist())
    test_cycles = set(df.loc[X_test.index, "cycle"].astype(str).unique().tolist())

    assert train_cycles == plan.train_cycles
    assert test_cycles == plan.test_cycles
    assert train_cycles.isdisjoint(test_cycles)


def test_choose_time_split_cycles_requires_multiple_cycles() -> None:
    df = pd.DataFrame({"cycle": ["2017-2018"] * 3, "ridageyr": [40, 50, 60], "diabetes": [0, 1, 0]})

    with pytest.raises(ValueError):
        choose_time_split_cycles(df)