from fastapi import APIRouter

from backend.app.schemas.overview import OverviewResponse
from backend.app.services.overview_service import build_overview_payload

overview_router = APIRouter()


@overview_router.get('/overview', response_model=OverviewResponse)
def overview() -> dict:
    return build_overview_payload()
