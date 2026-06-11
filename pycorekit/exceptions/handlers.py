from fastapi import Request
from fastapi.responses import JSONResponse
from .base import AppException
from pycorekit.logging.logger import get_logger

log = get_logger("exception")


async def app_exception_handler(request: Request, exc: AppException):
    """
    Handles known application exceptions raised intentionally
    using AppException or its subclasses.
    """
    log.error(f"AppException: {exc.message}")

    correlation_id = getattr(request.state, "correlation_id", None)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "error_type": exc.error_type,
            "correlation_id": correlation_id
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handles all unexpected exceptions that are NOT AppException.
    Ensures:
    - Logging
    - Correlation ID propagation
    - Consistent JSON response
    """
    log.error(f"Unhandled exception: {exc}", exc_info=True)

    correlation_id = getattr(request.state, "correlation_id", None)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "correlation_id": correlation_id
        }
    )
