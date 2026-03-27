from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.schemas.predict import PredictRequest


class ExplainRequest(BaseModel):
    task: str = Field(..., description="Task name, e.g. diabetes")
    input: PredictRequest
    top_k: int = 10


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class ExplainResponse(BaseModel):
    task: str
    probability: float
    base_value: float | None = None
    contributions: list[FeatureContribution]