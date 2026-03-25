from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["ridageyr", "lbxglu", "lbxtr", "lbdhdd", "lbxins", "bmxbmi"]


def apply_inclusion_rules(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["ridageyr"].ge(20)
    mask &= df[REQUIRED_COLUMNS].notna().all(axis=1)
    return df.loc[mask].copy()
