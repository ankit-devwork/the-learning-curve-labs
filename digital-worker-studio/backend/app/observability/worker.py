from typing import Optional
from .logger import logger, get_current_correlation_id


def get_worker_logger():
    """Fetches a contextual logger bound to the runtime context automatically."""
    cid = get_current_correlation_id()
    return logger.bind(correlation_id=cid) if cid else logger


def propagate_correlation_id(headers: Optional[dict] = None) -> dict:
    """Attaches tracing data explicitly onto outgoing connection interfaces."""
    headers = headers or {}
    cid = get_current_correlation_id()
    if cid:
        headers["x-correlation-id"] = cid
    return headers
