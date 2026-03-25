import pandas as pd

from chronic_disease_risk.preprocessing.splits import split_by_cycle


def test_split_by_cycle_separates_train_and_test_cycles() -> None:
    df = pd.DataFrame(
        {
            "cycle": ["2013-2014", "2015-2016", "2017-2018"],
            "value": [1, 2, 3],
        }
    )

    train_df, test_df = split_by_cycle(
        df,
        train_cycles={"2013-2014", "2015-2016"},
        test_cycles={"2017-2018"},
    )

    assert train_df["value"].tolist() == [1, 2]
    assert test_df["value"].tolist() == [3]
