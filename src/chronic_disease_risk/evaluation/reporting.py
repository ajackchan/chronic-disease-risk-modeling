from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_metrics_table(metrics: dict[str, float], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(destination, index=False)
