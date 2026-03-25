from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronic_disease_risk.preprocessing.nhanes_merge import merge_nhanes_tables


if __name__ == "__main__":
    merged = merge_nhanes_tables({})
    print(f"merged_rows={len(merged)}")
