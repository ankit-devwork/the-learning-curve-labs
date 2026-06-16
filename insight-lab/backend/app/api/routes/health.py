from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pycorekit.tracing.decorators import with_observability

from app.services.readiness import run_readiness_checks

router = APIRouter()


@router.get("/health")
@with_observability("health_check")
async def health(request: Request):
    correlation_id = getattr(request.state, "correlation_id", None)
    return {
        "status": "ok",
        "service": "insightlab-api",
        "correlation_id": correlation_id,
    }


@router.get("/ready")
@with_observability("readiness_check")
async def ready(request: Request):
    payload, status_code = await run_readiness_checks()
    correlation_id = getattr(request.state, "correlation_id", None)
    payload["correlation_id"] = correlation_id
    return JSONResponse(content=payload, status_code=status_code)
