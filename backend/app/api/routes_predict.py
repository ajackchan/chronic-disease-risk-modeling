from fastapi import APIRouter

from backend.app.schemas.predict import PredictRequest, PredictResponse
from backend.app.services.prediction_service import predict_all_tasks

predict_router = APIRouter()


@predict_router.post('/predict', response_model=PredictResponse)
def predict(payload: PredictRequest) -> dict:
    return {'predictions': predict_all_tasks(payload)}
