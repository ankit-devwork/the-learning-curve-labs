from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.services.readiness import run_readiness_checks

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "insightlab-api"}


@router.get("/ready")
async def ready():
    payload, status_code = await run_readiness_checks()
    return JSONResponse(content=payload, status_code=status_code)
