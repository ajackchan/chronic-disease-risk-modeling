from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronic_disease_risk.data_sources.nhanes_download import download_from_config


if __name__ == "__main__":
    download_from_config(config_path=REPO_ROOT / "configs" / "nhanes.yaml", repo_root=REPO_ROOT)
