from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes_health import health_router
from backend.app.api.routes_overview import overview_router
from backend.app.api.routes_predict import predict_router
from backend.app.api.routes_samples import samples_router
from backend.app.api.routes_tasks import tasks_router
from backend.app.config import REPO_ROOT


def create_app() -> FastAPI:
    app = FastAPI(title='Chronic Disease Risk Demo API')
    app.include_router(health_router, prefix='/api')
    app.include_router(overview_router, prefix='/api')
    app.include_router(tasks_router, prefix='/api')
    app.include_router(samples_router, prefix='/api')
    app.include_router(predict_router, prefix='/api')

    # Serve training artifacts (ROC plots, CSV summaries, etc.) from repo root.
    app.mount('/static', StaticFiles(directory=REPO_ROOT), name='static')

    # If the frontend has been built, serve it at '/'
    frontend_dist = Path(REPO_ROOT) / 'frontend' / 'dist'
    if frontend_dist.exists():
        app.mount('/', StaticFiles(directory=frontend_dist, html=True), name='frontend')

    return app