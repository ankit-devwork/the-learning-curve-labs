# tracing/middleware.py

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from pycorekit.correlation.context import (
    set_current_correlation_id,
    clear_correlation_id,
)
from pycorekit.correlation.headers import (
    resolve_request_correlation_id,
    tracking_response_headers,
)
from pycorekit.core_logging.logger import logger
from pycorekit.tracing.tracing import start_trace, get_current_trace, init_empty_trace


class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cid = resolve_request_correlation_id(request)

        request.state.correlation_id = cid
        set_current_correlation_id(cid)

        # ALWAYS initialize trace early
        request.state.trace = init_empty_trace()

        log = logger.bind(correlation_id=cid)
        start = time.perf_counter()

        log.info(f"Request started: {request.method} {request.url.path}")

        # Top-level HTTP trace
        with start_trace(
            name=f"{request.method} {request.url.path}",
            inputs={"path": request.url.path},
        ):
            try:
                response = await call_next(request)
            finally:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                log.info(
                    f"Request completed: {request.method} {request.url.path}",
                    duration_ms=duration_ms,
                )
                clear_correlation_id()

        # After all spans complete, attach sanitized trace
        raw_trace = get_current_trace() or request.state.trace
        request.state.trace = raw_trace

        for header_name, header_value in tracking_response_headers(cid).items():
            response.headers[header_name] = header_value
        return response
