from fastapi import APIRouter, Request
from pycorekit.tracing.decorators import with_observability
from pycorekit.tracing.tracing import start_trace
from pycorekit.correlation.context import get_current_correlation_id
from app.core.settings import settings

router = APIRouter(tags=["Observability"])


@router.get("/observability")
@with_observability("observability_test")
async def observability_test(request: Request):
    cid = get_current_correlation_id() or "unknown"

    with start_trace("observability_test_handler", inputs={"correlation_id": cid}):
        return {
            "status": "ok",
            "message": "Observability test OK",
            "correlation_id": cid,
            "model": settings.models.llm_model,
            "answer": None,
        }
