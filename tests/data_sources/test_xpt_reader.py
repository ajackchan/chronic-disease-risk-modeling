import pandas as pd

from chronic_disease_risk.data_sources.xpt_reader import normalize_columns


def test_normalize_columns_lowercases_and_strips_names() -> None:
    df = pd.DataFrame(columns=["SEQN", "RIDAGEYR "])

    normalized = normalize_columns(df)

    assert list(normalized.columns) == ["seqn", "ridageyr"]
