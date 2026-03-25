from __future__ import annotations

from chronic_disease_risk.preprocessing.nhanes_merge import merge_nhanes_tables


if __name__ == "__main__":
    merged = merge_nhanes_tables({})
    print(f"merged_rows={len(merged)}")
