from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronic_disease_risk.modeling.candidates import build_candidate_models


if __name__ == "__main__":
    models = build_candidate_models(random_state=42)
    print("candidate_models=", ",".join(models))
