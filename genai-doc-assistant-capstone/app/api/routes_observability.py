from fastapi import APIRouter, Request
from pycorekit.tracing.decorators import with_observability
from pycorekit.tracing.tracing import start_trace
from pycorekit.correlation.context import get_current_correlation_id

from app.core.settings import settings
from pycorekit.utils.sanitize_observability import sanitize_observability   # <-- add this

router = APIRouter(tags=["Observability"])


@router.get("/observability")
@with_observability("observability_test")
async def observability_test(request: Request):
    cid = get_current_correlation_id() or "unknown"

    with start_trace("observability_test_handler", inputs={"correlation_id": cid}):

        trace = getattr(request.state, "trace", None)

        if trace is None:
            trace = {
                "spans": [],
                "errors": ["trace_missing_in_request_state"],
            }

        
        safe_trace = sanitize_observability(trace)

        return {
            "status": "ok",
            "message": "Observability test OK",
            "correlation_id": cid,
            "model": None,
            "answer": None,
            "observability": safe_trace,
        }
