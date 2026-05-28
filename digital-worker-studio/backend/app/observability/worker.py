from typing import Optional
from .logger import logger, get_current_correlation_id, generate_correlation_id


def get_worker_logger():
    """Fetches a contextual logger bound to the runtime context automatically."""
    return logger.bind(correlation_id=get_current_correlation_id())


def propagate_correlation_id(headers: Optional[dict] = None) -> dict:
    """Attaches tracing data explicitly onto outgoing connection interfaces."""
    headers = headers or {}
    headers["x-correlation-id"] = get_current_correlation_id()
    return headers