from contextlib import contextmanager
from typing import Dict, Any, Optional
from langfuse import get_client
from app.observability.logger import logger

# Initialize client cleanly via standard singleton patterns
langfuse = get_client()


@contextmanager
def start_trace(span_name: str, correlation_id: str) -> Dict[str, Optional[Any]]:
    """Langfuse v4 context tracking wrapper block."""
    lf_cm = None
    lf_span = None
    ls_span = None

    try:
        lf_cm = langfuse.start_as_current_observation(
            as_type="span",
            name=span_name,
            metadata={"correlation_id": correlation_id},
        )
        lf_span = lf_cm.__enter__()

        try:
            from langsmith import trace as ls_trace
            ls_span = ls_trace(span_name)
        except Exception:
            ls_span = None

        yield {
            "langfuse": lf_span,
            "langsmith": ls_span,
        }

    except Exception as e:
        logger.exception("Tracing initialization error", error=str(e))
        raise

    finally:
        if ls_span:
            try:
                ls_span.end()
            except Exception:
                pass

        if lf_cm:
            try:
                lf_cm.__exit__(None, None, None)
            except Exception:
                pass

        try:
            langfuse.flush()
        except Exception:
            pass