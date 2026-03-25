from __future__ import annotations

import pandas as pd


def split_by_cycle(
    df: pd.DataFrame,
    train_cycles: set[str],
    test_cycles: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["cycle"].isin(train_cycles)].copy()
    test_df = df[df["cycle"].isin(test_cycles)].copy()
    return train_df, test_df
