from __future__ import annotations

import pandas as pd

INSULIN_CANDIDATES = ["lbxin", "lbxins"]


def _resolve_insulin_column(df: pd.DataFrame) -> str:
    for column in INSULIN_CANDIDATES:
        if column in df.columns:
            return column
    raise KeyError("No insulin column found; expected one of lbxin or lbxins")


def apply_inclusion_rules(df: pd.DataFrame) -> pd.DataFrame:
    insulin_column = _resolve_insulin_column(df)

    # For GLM7 and derived features we need these core measurements.
    required_columns = [
        "ridageyr",
        "bmxbmi",
        "lbxglu",
        "lbxtr",
        "lbdhdd",
        "lbdldl",
        insulin_column,
    ]

    mask = df["ridageyr"].ge(20)
    mask &= df[required_columns].notna().all(axis=1)

    # Keep only examined participants when available.
    if "ridstatr" in df.columns:
        mask &= df["ridstatr"].eq(2)

    return df.loc[mask].copy()