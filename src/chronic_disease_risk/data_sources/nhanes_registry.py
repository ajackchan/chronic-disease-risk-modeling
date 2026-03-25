from __future__ import annotations

BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"


def build_xpt_url(cycle_suffix: str, component: str, table: str) -> str:
    return f"{BASE_URL}/{component}/{table}_{cycle_suffix}.XPT"
