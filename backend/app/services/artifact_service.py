from __future__ import annotations

from pathlib import Path

from backend.app.config import REPORTS_DIR


def build_task_artifact_payload(task_name: str, reports_dir: Path = REPORTS_DIR) -> dict:
    return {
        'task': task_name,
        'roc_plot_url': f'/static/reports/tables/baseline_{task_name}_roc.png',
        'confusion_matrix_url': f'/static/reports/tables/baseline_{task_name}_confusion.png',
        'comparison_plot_url': f'/static/reports/tables/candidate_{task_name}_comparison.png',
        'summary_csv_url': f'/static/reports/tables/candidate_{task_name}_summary.csv',
    }
