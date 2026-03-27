from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplitPlan:
    train_cycles: set[str]
    test_cycles: set[str]


_CYCLE_YEAR_RE = re.compile(r"(?P<start>\d{4})")


def _cycle_start_year(cycle: str) -> int:
    match = _CYCLE_YEAR_RE.search(str(cycle))
    if not match:
        raise ValueError(f"Invalid cycle label: {cycle!r}")
    return int(match.group("start"))


def choose_time_split_cycles(df: pd.DataFrame, *, test_size: float = 0.25) -> TimeSplitPlan:
    """Pick train/test cycles for a time-based split.

    Strategy:
    - Sort cycles by start year.
    - Take the latest cycles as test until reaching `test_size` of samples.

    Raises ValueError if there are fewer than 2 distinct cycles.
    """

    if "cycle" not in df.columns:
        raise ValueError("Missing column 'cycle' required for time-based split")

    cycles = [str(c) for c in df["cycle"].dropna().unique().tolist()]
    if len(cycles) < 2:
        raise ValueError(
            "Time-based split requires at least 2 NHANES cycles in the dataset. "
            "Your current dataset only contains one cycle. "
            "Add more cycles in configs/nhanes.yaml, re-download, and rebuild the dataset."
        )

    cycles_sorted = sorted(cycles, key=_cycle_start_year)

    counts = df["cycle"].astype(str).value_counts().to_dict()
    total = float(len(df))

    test_cycles: list[str] = []
    test_count = 0.0

    # Accumulate from the latest cycle backwards.
    for cycle in reversed(cycles_sorted):
        # Keep at least 1 cycle for train.
        if len(test_cycles) >= len(cycles_sorted) - 1:
            break

        test_cycles.append(cycle)
        test_count += float(counts.get(cycle, 0))

        if total > 0 and (test_count / total) >= float(test_size):
            break

    test_set = set(test_cycles)
    train_set = set(cycles_sorted) - test_set

    if not train_set or not test_set:
        raise ValueError(
            f"Failed to build time split: train_cycles={sorted(train_set)}, test_cycles={sorted(test_set)}"
        )

    return TimeSplitPlan(train_cycles=train_set, test_cycles=test_set)


def split_train_test_by_time(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    test_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, TimeSplitPlan]:
    """Return X_train, X_test, y_train, y_test, plan using cycle-based time split."""

    plan = choose_time_split_cycles(df, test_size=test_size)

    train_df = df[df["cycle"].astype(str).isin(plan.train_cycles)].copy()
    test_df = df[df["cycle"].astype(str).isin(plan.test_cycles)].copy()

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    return X_train, X_test, y_train, y_test, plan