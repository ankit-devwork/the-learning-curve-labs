"""
Decorator for:
- Logging spans
- Measuring duration
- Optional telemetry
"""

import time
import asyncio
from functools import wraps
from typing import Callable, Any, Optional

from pycorekit.logging.logger import logger
from pycorekit.correlation.context import get_current_correlation_id
from pycorekit.tracing.tracing import start_trace

def with_observability(name: Optional[str] = None):
    def decorator(func: Callable):
        func_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cid = get_current_correlation_id()
            bound = logger.bind(correlation_id=cid, span=func_name)

            bound.info("Span started")
            with start_trace(func_name, cid or "unknown"):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = round((time.time() - start) * 1000, 2)
                    bound.info("Span completed", duration_ms=duration)
                    return result
                except Exception:
                    bound.exception("Span failed")
                    raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else func

    return decorator
