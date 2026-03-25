from __future__ import annotations

from pydantic import BaseModel


class TaskOverview(BaseModel):
    task: str
    best_model_name: str
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class OverviewResponse(BaseModel):
    title: str
    tasks: list[TaskOverview]
