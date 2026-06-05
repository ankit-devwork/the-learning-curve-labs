"""
FastAPI middleware that:
- Extracts or generates correlation ID
- Stores it in ContextVar
- Adds it to logs
- Adds it to response headers
"""

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from pycorekit.logging.logger import logger
from pycorekit.correlation.context import (
    generate_correlation_id,
    correlation_id_ctx,
    set_correlation_id,
)

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()

        cid = request.headers.get("x-correlation-id", generate_correlation_id())
        token = correlation_id_ctx.set(cid)
        set_correlation_id(cid)
        request.state.correlation_id = cid

        bound = logger.bind(correlation_id=cid)
        bound.info("Incoming request", method=request.method, path=request.url.path)

        try:
            response = await call_next(request)
            duration = round((time.time() - start) * 1000, 2)

            bound.info("Request completed", status_code=response.status_code, duration_ms=duration)
            response.headers["x-correlation-id"] = cid
            return response

        except Exception as exc:
            bound.exception("Unhandled exception during request")
            raise exc

        finally:
            correlation_id_ctx.reset(token)
