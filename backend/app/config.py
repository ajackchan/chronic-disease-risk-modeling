from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CONFIGS_DIR = REPO_ROOT / 'configs'
REPORTS_DIR = REPO_ROOT / 'reports' / 'tables'
ARTIFACTS_DIR = REPO_ROOT / 'artifacts'
