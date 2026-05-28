from fastapi import Request
from fastapi.responses import JSONResponse
from app.observability.logger import logger, get_current_correlation_id
from app.core.exceptions import AppException


async def app_exception_handler(request: Request, exc: AppException):
    cid = get_current_correlation_id()
    bound = logger.bind(correlation_id=cid)

    bound.error(
        exc.message,
        details=exc.details,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "correlation_id": cid,
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    cid = get_current_correlation_id()
    bound = logger.bind(correlation_id=cid)

    bound.exception("Unhandled server error encountered")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "correlation_id": cid,
        }
    )