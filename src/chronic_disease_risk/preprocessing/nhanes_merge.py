from __future__ import annotations

import pandas as pd


def merge_nhanes_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for _, frame in tables.items():
        frame = frame.drop_duplicates(subset=["seqn"])
        merged = frame if merged is None else merged.merge(frame, on="seqn", how="inner")

    if merged is None:
        return pd.DataFrame()

    return merged
