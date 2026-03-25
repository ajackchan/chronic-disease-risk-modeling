from __future__ import annotations

from pydantic import BaseModel

from backend.app.schemas.overview import TaskOverview


class TaskArtifactsResponse(BaseModel):
    task: str
    roc_plot_url: str
    confusion_matrix_url: str
    comparison_plot_url: str
    summary_csv_url: str


class TasksResponse(BaseModel):
    tasks: list[TaskOverview]