from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_run_nhanes_pipeline_script_creates_outputs(repo_root: Path) -> None:
    artifact = repo_root / "artifacts" / "smoke_task.json"
    metrics = repo_root / "reports" / "tables" / "smoke_metrics.csv"

    if artifact.exists():
        artifact.unlink()
    if metrics.exists():
        metrics.unlink()

    result = subprocess.run(
        [sys.executable, "scripts/run_nhanes_pipeline.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert artifact.exists()
    assert metrics.exists()
