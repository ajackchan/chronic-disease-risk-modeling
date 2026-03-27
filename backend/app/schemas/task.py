from __future__ import annotations

from pydantic import BaseModel

from backend.app.schemas.overview import TaskOverview


class TaskArtifactsResponse(BaseModel):
    task: str

    # Backward compatible fields (already used by the frontend).
    roc_plot_url: str
    confusion_matrix_url: str
    comparison_plot_url: str
    summary_csv_url: str

    # Extra artifacts for thesis / defense. These may be None if the file is not generated yet.
    baseline_calibration_url: str | None = None
    baseline_dca_url: str | None = None

    candidate_best_roc_plot_url: str | None = None
    candidate_best_confusion_matrix_url: str | None = None
    candidate_best_calibration_url: str | None = None
    candidate_best_dca_url: str | None = None

    candidate_best_shap_summary_url: str | None = None
    xgboost_tuned_params_url: str | None = None


class TasksResponse(BaseModel):
    tasks: list[TaskOverview]