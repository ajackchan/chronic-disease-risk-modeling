from fastapi import APIRouter

from backend.app.schemas.task import TaskArtifactsResponse, TasksResponse
from backend.app.services.artifact_service import build_task_artifact_payload
from backend.app.services.overview_service import get_task_names, load_task_overview

tasks_router = APIRouter()


@tasks_router.get('/tasks', response_model=TasksResponse)
def tasks() -> dict:
    task_payloads = [load_task_overview(task_name) for task_name in get_task_names()]
    return {'tasks': task_payloads}


@tasks_router.get('/tasks/{task_name}/artifacts', response_model=TaskArtifactsResponse)
def task_artifacts(task_name: str) -> dict:
    return build_task_artifact_payload(task_name)