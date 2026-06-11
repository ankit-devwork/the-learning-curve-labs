# tracing/tracing.py

import time
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any
import contextvars

from pycorekit.logging.logger import logger
from pycorekit.correlation.context import get_current_correlation_id

_TRACE_CTX: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "TRACE_CTX", default=None
)

# Lazy-initialized clients
_langfuse_client = None
_langsmith_client = None


def _get_langfuse_client():
    """Lazy initialization of Langfuse client with diagnostics."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    try:
        if not os.getenv("LANGFUSE_SECRET_KEY"):
            logger.warning("LANGFUSE_SECRET_KEY not set; Langfuse will be disabled")
            return None
        if not os.getenv("LANGFUSE_PUBLIC_KEY"):
            logger.warning("LANGFUSE_PUBLIC_KEY not set; Langfuse will be disabled")
            return None

        from langfuse import get_client as get_langfuse
        _langfuse_client = get_langfuse()
        logger.info("Langfuse client initialized successfully")
        return _langfuse_client
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse: {e}")
        return None


def _get_langsmith_client():
    """Lazy initialization of LangSmith client with diagnostics."""
    global _langsmith_client
    if _langsmith_client is not None:
        return _langsmith_client

    try:
        if not os.getenv("LANGCHAIN_API_KEY"):
            logger.warning("LANGCHAIN_API_KEY not set; LangSmith will be disabled")
            return None

        from langsmith import Client as LangSmithClient
        _langsmith_client = LangSmithClient()
        logger.info("LangSmith client initialized successfully")
        return _langsmith_client
    except Exception as e:
        logger.warning(f"Failed to initialize LangSmith: {e}")
        return None


def init_empty_trace():
    """
    Create a fresh trace structure and store it in the current context.

    Returns:
        dict: Trace scaffold containing `langfuse`, `langsmith`, `spans`, and `errors`.

    Example:
        trace = init_empty_trace()
        # trace is now available through get_current_trace()
    """
    trace = {
        "langfuse": None,
        "langsmith": None,
        "spans": [],
        "errors": []
    }
    _TRACE_CTX.set(trace)
    return trace


def get_current_trace() -> Dict[str, Any] | None:
    """
    Retrieve the current trace from context.

    Returns:
        dict or None: The current trace object, or None if no trace is initialized.

    Example:
        trace = get_current_trace()
    """
    return _TRACE_CTX.get()


def _init_trace(name: str, inputs: dict | None) -> Dict[str, Any]:
    cid = get_current_correlation_id() or "unknown"

    trace = init_empty_trace()

    # Langfuse
    langfuse = _get_langfuse_client()
    if langfuse:
        try:
            trace["langfuse"] = langfuse.start_observation(
                name=name,
                metadata={"correlation_id": cid}
            )
        except Exception as e:
            logger.warning("Langfuse start_observation failed", error=str(e))
            trace["errors"].append(f"langfuse_start_failed: {e}")

    # LangSmith
    langsmith = _get_langsmith_client()
    if langsmith:
        try:
            trace["langsmith"] = langsmith.create_run(
                name=name,
                run_type="chain",
                inputs=inputs or {"input": "<none>"},
                metadata={"correlation_id": cid}
            )
        except Exception as e:
            logger.warning("LangSmith run creation failed", error=str(e))
            trace["errors"].append(f"langsmith_start_failed: {e}")

    return trace


@contextmanager
def start_trace(name: str, inputs: Optional[dict] = None):
    """
    Start a trace span.

    If there is no active trace in context, this creates a new one.
    The returned context manager yields langfuse/langsmith handles and the span.

    Example:
        with start_trace("process_data", inputs={"batch_id": 123}):
            do_work()
    """
    trace = get_current_trace()
    if trace is None:
        trace = _init_trace(name, inputs)

    lf_obs = trace.get("langfuse")
    ls_run = trace.get("langsmith")

    span = {
        "name": name,
        "inputs": inputs or {},
        "start_ts": time.perf_counter(),
        "end_ts": None,
        "duration_ms": None,
        "error": None,
    }
    trace["spans"].append(span)

    try:
        yield {"langfuse": lf_obs, "langsmith": ls_run, "span": span}
    except Exception as e:
        span["error"] = str(e)
        raise
    finally:
        span["end_ts"] = time.perf_counter()
        span["duration_ms"] = round(
            (span["end_ts"] - span["start_ts"]) * 1000, 2
        )
