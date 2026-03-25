from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# pytest in this repo is configured with pythonpath=['src']; ensure repo root is also importable
# so `import backend...` works when running `pytest backend/tests`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.main import create_app  # noqa: E402


def create_test_client() -> TestClient:
    return TestClient(create_app())