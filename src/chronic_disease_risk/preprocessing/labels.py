from __future__ import annotations

import pandas as pd


def add_multimorbidity_label(df: pd.DataFrame, outcome_columns: list[str]) -> pd.DataFrame:
    labeled = df.copy()
    labeled["multimorbidity_count"] = labeled[outcome_columns].sum(axis=1)
    labeled["multimorbidity_label"] = (labeled["multimorbidity_count"] >= 2).astype(int)
    return labeled
