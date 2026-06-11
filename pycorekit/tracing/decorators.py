# tracing/decorators.py

import time
import asyncio
from functools import wraps
from typing import Callable, Optional

from langsmith import Client as LangSmithClient

from pycorekit.logging.logger import logger
from pycorekit.correlation.context import get_current_correlation_id
from pycorekit.tracing.tracing import start_trace, get_current_trace
from pycorekit.utils.sanitize_observability import sanitize_observability

langsmith = LangSmithClient()


# ---------------------------------------------------------
# SAFE INPUT SANITIZER
# ---------------------------------------------------------
def _safe_inputs(args, kwargs):
    """
    Convert args/kwargs into JSON‑safe primitives.
    Removes Request objects and any non‑serializable values.
    """
    clean = {}

    # Sanitize args
    clean_args = []
    for a in args:
        # Detect Starlette/FastAPI Request object
        if hasattr(a, "scope") and hasattr(a, "url"):
            clean_args.append("<Request>")
        else:
            clean_args.append(str(a))
    clean["args"] = clean_args

    # Sanitize kwargs
    clean_kwargs = {}
    for k, v in kwargs.items():
        if hasattr(v, "scope") and hasattr(v, "url"):
            clean_kwargs[k] = "<Request>"
        else:
            clean_kwargs[k] = str(v)
    clean["kwargs"] = clean_kwargs

    return clean


# ---------------------------------------------------------
# MAIN DECORATOR
# ---------------------------------------------------------
def _find_request(args, kwargs):
    for arg in args:
        if hasattr(arg, "scope") and hasattr(arg, "url"):
            return arg
    for value in kwargs.values():
        if hasattr(value, "scope") and hasattr(value, "url"):
            return value
    return None


def with_observability(name: Optional[str] = None):
    """
    Decorator for FastAPI routes that creates an observability trace.

    Behavior:
    - starts a top-level observation span for the route
    - updates Langfuse/LangSmith metadata on success or failure
    - after the route returns, injects a sanitized trace into dict results
      under the `observability` key

    This ensures the outer route span is complete before the trace is serialized.

    Example:
        @router.post("/upload")
        @with_observability("upload_and_ingest")
        async def upload(request: Request):
            return {"status": "ok"}

        # Response will include `observability` when returning a dict.
    """

    def decorator(func: Callable):
        func_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cid = get_current_correlation_id() or "unknown"
            bound = logger.bind(correlation_id=cid, observation=func_name)

            bound.info("Observation started")
            start = time.perf_counter()

            # 🔥 SAFE INPUTS (no Request objects)
            trace_inputs = _safe_inputs(args, kwargs)

            # ---------------------------------------------------------
            # Start span
            # ---------------------------------------------------------
            with start_trace(func_name, inputs=trace_inputs) as obs:
                lf_obs = obs.get("langfuse")
                ls_run = obs.get("langsmith")

                try:
                    # -------------------------------------------------
                    # Execute wrapped function
                    # -------------------------------------------------
                    result = await func(*args, **kwargs)

                    duration_ms = round((time.perf_counter() - start) * 1000, 2)
                    bound.info("Observation completed", duration_ms=duration_ms)

                    # -------------------------------------------------
                    # Langfuse update
                    # -------------------------------------------------
                    if lf_obs:
                        try:
                            lf_obs.update(
                                output={"status": "success"},
                                metadata={"duration_ms": duration_ms}
                            )
                        except Exception as e:
                            bound.warning("Langfuse update failed", error=str(e))

                    # -------------------------------------------------
                    # LangSmith update
                    # -------------------------------------------------
                    if ls_run:
                        try:
                            langsmith.update_run(
                                ls_run["id"],
                                metadata={"duration_ms": duration_ms},
                                outputs={"result": str(result)}
                            )
                        except Exception as e:
                            bound.warning("LangSmith update failed", error=str(e))

                    if isinstance(result, dict):
                        request = _find_request(args, kwargs)
                        trace = None
                        if request is not None:
                            trace = getattr(request.state, "trace", None)
                        if trace is None:
                            trace = get_current_trace()

                        if trace is not None:
                            result["observability"] = sanitize_observability(trace)

                    return result

                except Exception as e:
                    duration_ms = round((time.perf_counter() - start) * 1000, 2)
                    bound.exception("Observation failed", duration_ms=duration_ms)

                    # -------------------------------------------------
                    # Langfuse error update
                    # -------------------------------------------------
                    if lf_obs:
                        try:
                            lf_obs.update(
                                output={"error": str(e)},
                                metadata={"duration_ms": duration_ms}
                            )
                        except Exception:
                            pass

                    # -------------------------------------------------
                    # LangSmith error update
                    # -------------------------------------------------
                    if ls_run:
                        try:
                            langsmith.update_run(
                                ls_run["id"],
                                error=str(e),
                                metadata={"duration_ms": duration_ms}
                            )
                        except Exception:
                            pass

                    raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else func

    return decorator
