# Lightweight observability package surface: avoid importing heavy submodules at package import
from .logger import logger, generate_correlation_id
from .middleware import ObservabilityMiddleware

__all__ = [
    "logger",
    "generate_correlation_id",
    "ObservabilityMiddleware",
]
