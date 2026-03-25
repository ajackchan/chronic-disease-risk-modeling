from __future__ import annotations

from pydantic import BaseModel


class PredictRequest(BaseModel):
    ridageyr: float
    bmxbmi: float
    lbxglu: float
    lbxtr: float
    lbdhdd: float
    lbdldl: float
    lbxin: float


class TaskPrediction(BaseModel):
    probability: float
    risk_label: str


class PredictResponse(BaseModel):
    predictions: dict[str, TaskPrediction]
