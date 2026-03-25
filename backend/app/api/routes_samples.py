from fastapi import APIRouter

from backend.app.services.sample_service import get_demo_samples

samples_router = APIRouter()


@samples_router.get('/samples')
def samples() -> dict:
    return {'samples': get_demo_samples()}
