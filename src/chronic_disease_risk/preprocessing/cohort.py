from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["ridageyr", "lbxglu", "lbxtr", "lbdhdd", "insulin", "bmxbmi"]
INSULIN_CANDIDATES = ["lbxin", "lbxins"]


def _resolve_insulin_column(df: pd.DataFrame) -> str:
    for column in INSULIN_CANDIDATES:
        if column in df.columns:
            return column
    raise KeyError("No insulin column found; expected one of lbxin or lbxins")


def apply_inclusion_rules(df: pd.DataFrame) -> pd.DataFrame:
    insulin_column = _resolve_insulin_column(df)
    required_columns = ["ridageyr", "lbxglu", "lbxtr", "lbdhdd", insulin_column, "bmxbmi"]

    mask = df["ridageyr"].ge(20)
    mask &= df[required_columns].notna().all(axis=1)
    if "ridstatr" in df.columns:
        mask &= df["ridstatr"].eq(2)
    return df.loc[mask].copy()
