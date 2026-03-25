from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: column.strip().lower() for column in df.columns}
    return df.rename(columns=renamed)


def read_xpt(path: Path) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport", encoding="utf-8")
    return normalize_columns(df)
