from fastapi import APIRouter, HTTPException

from backend.app.schemas.explain import ExplainRequest, ExplainResponse
from backend.app.services.explain_service import explain_single_task

explain_router = APIRouter()


@explain_router.post('/explain', response_model=ExplainResponse)
def explain(payload: ExplainRequest) -> dict:
    try:
        data = explain_single_task(payload.task, payload.input, top_k=payload.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # explain_service returns extra keys (risk_label) that are ignored by response_model.
    return data