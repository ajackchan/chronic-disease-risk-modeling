from __future__ import annotations

from pathlib import Path

from backend.app.config import REPORTS_DIR


def _reports_url(filename: str) -> str:
    return f"/static/reports/tables/{filename}"


def _reports_url_if_exists(reports_dir: Path, filename: str) -> str | None:
    path = reports_dir / filename
    if path.exists() and path.is_file():
        return _reports_url(filename)
    return None


def build_task_artifact_payload(task_name: str, reports_dir: Path = REPORTS_DIR) -> dict:
    # Keep these stable for the current frontend.
    payload = {
        'task': task_name,
        'roc_plot_url': _reports_url(f'baseline_{task_name}_roc.png'),
        'confusion_matrix_url': _reports_url(f'baseline_{task_name}_confusion.png'),
        'comparison_plot_url': _reports_url(f'candidate_{task_name}_comparison.png'),
        'summary_csv_url': _reports_url(f'candidate_{task_name}_summary.csv'),
    }

    # Extra artifacts (may not exist yet).
    payload.update(
        {
            'baseline_calibration_url': _reports_url_if_exists(reports_dir, f'baseline_{task_name}_calibration.png'),
            'baseline_dca_url': _reports_url_if_exists(reports_dir, f'baseline_{task_name}_dca.png'),
            'candidate_best_roc_plot_url': _reports_url_if_exists(reports_dir, f'candidate_best_{task_name}_roc.png'),
            'candidate_best_confusion_matrix_url': _reports_url_if_exists(
                reports_dir, f'candidate_best_{task_name}_confusion.png'
            ),
            'candidate_best_calibration_url': _reports_url_if_exists(
                reports_dir, f'candidate_best_{task_name}_calibration.png'
            ),
            'candidate_best_dca_url': _reports_url_if_exists(reports_dir, f'candidate_best_{task_name}_dca.png'),
            'candidate_best_shap_summary_url': _reports_url_if_exists(
                reports_dir, f'candidate_best_{task_name}_shap.png'
            ),
            'xgboost_tuned_params_url': _reports_url_if_exists(
                reports_dir, f'candidate_{task_name}_xgboost_tuned_params.json'
            ),
        }
    )

    return payload