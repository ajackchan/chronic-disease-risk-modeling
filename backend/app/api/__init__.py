from .routes_health import health_router
from .routes_overview import overview_router
from .routes_predict import predict_router
from .routes_samples import samples_router
from .routes_tasks import tasks_router

__all__ = ['health_router', 'overview_router', 'tasks_router', 'samples_router', 'predict_router']
