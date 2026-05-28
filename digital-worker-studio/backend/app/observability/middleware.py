import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.observability.logger import logger, generate_correlation_id, correlation_id_ctx


class ObservabilityMiddleware(BaseHTTPMiddleware):
    
    async def dispatch(self, request: Request, call_next):
        start = time.time()

        # Extract or generate trace link
        correlation_id = request.headers.get("x-correlation-id", generate_correlation_id())
        
        # Lock variable into ContextVar lifecycle
        token = correlation_id_ctx.set(correlation_id)
        request.state.correlation_id = correlation_id

        # Bind context securely to loguru contextual processing
        bound = logger.bind(correlation_id=correlation_id)

        bound.info(
            "Incoming request",
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
            duration = round((time.time() - start) * 1000, 2)
            
            bound.info(
                "Request completed",
                status_code=response.status_code,
                duration_ms=duration,
            )
            
            response.headers["x-correlation-id"] = correlation_id
            return response

        except Exception as exc:
            duration = round((time.time() - start) * 1000, 2)
            bound.exception(
                "Unhandled exception during request",
                duration_ms=duration,
            )
            raise exc
            
        finally:
            # Clean context allocation after request stack fully unwinds
            correlation_id_ctx.reset(token)