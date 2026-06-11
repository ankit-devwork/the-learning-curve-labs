"""
Worker-safe logger for background tasks.
Automatically binds correlation ID if available.
"""

from pycorekit.core_logging.logger import logger
from pycorekit.correlation.context import get_current_correlation_id

def get_worker_logger():
    cid = get_current_correlation_id()
    return logger.bind(correlation_id=cid) if cid else logger
