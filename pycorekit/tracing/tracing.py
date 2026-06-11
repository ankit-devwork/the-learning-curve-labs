# tracing/tracing.py

import time
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any
import contextvars

from pycorekit.core_logging.logger import logger
from pycorekit.correlation.context import get_current_correlation_id

_TRACE_CTX: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "TRACE_CTX", default=None
)

# Lazy-initialized clients
_langfuse_client = None
_langsmith_client = None


def _langfuse_configured() -> bool:
    return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))


def _langsmith_configured() -> bool:
    if not os.getenv("LANGCHAIN_API_KEY"):
        return False
    tracing_flag = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    return tracing_flag in ("true", "1", "yes")


def get_external_tracing_status() -> Dict[str, Any]:
    """Return configured/active status for Langfuse and LangSmith."""
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    langsmith_project = os.getenv("LANGCHAIN_PROJECT", "genai-doc-assistant")
    langsmith_base = os.getenv("LANGSMITH_ENDPOINT", "https://smith.langchain.com").rstrip("/")

    return {
        "langfuse": {
            "configured": _langfuse_configured(),
            "host": langfuse_host,
        },
        "langsmith": {
            "configured": _langsmith_configured(),
            "project": langsmith_project,
            "base_url": langsmith_base,
            "url": f"{langsmith_base}/o/default/projects/p/{langsmith_project}",
        },
    }


def _get_langfuse_client():
    """Lazy initialization of Langfuse client with diagnostics."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    try:
        if not _langfuse_configured():
            return None

        os.environ.setdefault("LANGFUSE_HOST", "https://cloud.langfuse.com")
        from langfuse import get_client as get_langfuse
        _langfuse_client = get_langfuse()
        logger.info("Langfuse client initialized successfully")
        return _langfuse_client
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse: {e}")
        return None


def get_langsmith_client():
    """Lazy initialization of LangSmith client with diagnostics."""
    global _langsmith_client
    if _langsmith_client is not None:
        return _langsmith_client

    try:
        if not _langsmith_configured():
            return None

        os.environ.setdefault("LANGCHAIN_PROJECT", "genai-doc-assistant")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        from langsmith import Client as LangSmithClient
        _langsmith_client = LangSmithClient()
        logger.info("LangSmith client initialized successfully")
        return _langsmith_client
    except Exception as e:
        logger.warning(f"Failed to initialize LangSmith: {e}")
        return None


def _get_langsmith_client():
    return get_langsmith_client()


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


def _ensure_external_tracing(trace: Dict[str, Any], name: str, inputs: dict | None) -> None:
    """Attach Langfuse/LangSmith handles once per request trace."""
    if trace.get("_external_tracing_initialized"):
        return

    trace["_external_tracing_initialized"] = True
    cid = get_current_correlation_id() or "unknown"

    langfuse = _get_langfuse_client()
    if langfuse:
        try:
            trace["langfuse"] = langfuse.start_observation(
                name=name,
                metadata={"correlation_id": cid},
            )
        except Exception as e:
            logger.warning("Langfuse start_observation failed", error=str(e))
            trace["errors"].append(f"langfuse_start_failed: {e}")

    langsmith = _get_langsmith_client()
    if langsmith:
        try:
            trace["langsmith"] = langsmith.create_run(
                name=name,
                run_type="chain",
                inputs=inputs or {"input": "<none>"},
                metadata={"correlation_id": cid},
            )
        except Exception as e:
            logger.warning("LangSmith run creation failed", error=str(e))
            trace["errors"].append(f"langsmith_start_failed: {e}")


def _init_trace(name: str, inputs: dict | None) -> Dict[str, Any]:
    trace = init_empty_trace()
    _ensure_external_tracing(trace, name, inputs)
    return trace


@contextmanager
def start_trace(name: str, inputs: Optional[dict] = None, span_type: Optional[str] = None):
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
    else:
        _ensure_external_tracing(trace, name, inputs)

    lf_obs = trace.get("langfuse")
    ls_run = trace.get("langsmith")

    span = {
        "name": name,
        "type": span_type,
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
