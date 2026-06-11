# tracing/decorators.py

import time
import asyncio
from functools import wraps
from typing import Callable, Optional

from pycorekit.core_logging.logger import logger
from pycorekit.correlation.context import get_current_correlation_id
from pycorekit.tracing.tracing import start_trace, get_current_trace, get_langsmith_client
from pycorekit.utils.sanitize_observability import sanitize_observability


def _safe_inputs(args, kwargs):
    clean = {}
    clean_args = []
    for a in args:
        if hasattr(a, "scope") and hasattr(a, "url"):
            clean_args.append("<Request>")
        else:
            clean_args.append(str(a))
    clean["args"] = clean_args

    clean_kwargs = {}
    for k, v in kwargs.items():
        if hasattr(v, "scope") and hasattr(v, "url"):
            clean_kwargs[k] = "<Request>"
        else:
            clean_kwargs[k] = str(v)
    clean["kwargs"] = clean_kwargs

    return clean


def _find_request(args, kwargs):
    for arg in args:
        if hasattr(arg, "scope") and hasattr(arg, "url"):
            return arg
    for value in kwargs.values():
        if hasattr(value, "scope") and hasattr(value, "url"):
            return value
    return None


def _attach_observability(result, args, kwargs):
    if not isinstance(result, dict):
        return result

    request = _find_request(args, kwargs)
    trace = None
    if request is not None:
        trace = getattr(request.state, "trace", None)
    if trace is None:
        trace = get_current_trace()
    if trace is not None:
        result["observability"] = sanitize_observability(trace)
    return result


def with_observability(name: Optional[str] = None):
    def decorator(func: Callable):
        func_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cid = get_current_correlation_id() or "unknown"
            bound = logger.bind(correlation_id=cid, observation=func_name)
            langsmith = get_langsmith_client()

            bound.info("Observation started")
            start = time.perf_counter()
            trace_inputs = _safe_inputs(args, kwargs)

            result = None
            with start_trace(func_name, inputs=trace_inputs) as obs:
                lf_obs = obs.get("langfuse")
                ls_run = obs.get("langsmith")

                try:
                    result = await func(*args, **kwargs)

                    duration_ms = round((time.perf_counter() - start) * 1000, 2)
                    bound.info("Observation completed", duration_ms=duration_ms)

                    if lf_obs:
                        try:
                            lf_obs.update(
                                output={"status": "success"},
                                metadata={"duration_ms": duration_ms},
                            )
                        except Exception as e:
                            bound.warning("Langfuse update failed", error=str(e))

                    if ls_run and langsmith:
                        try:
                            langsmith.update_run(
                                ls_run["id"],
                                metadata={"duration_ms": duration_ms},
                                outputs={"result": str(result)},
                            )
                        except Exception as e:
                            bound.warning("LangSmith update failed", error=str(e))

                except Exception as e:
                    duration_ms = round((time.perf_counter() - start) * 1000, 2)
                    bound.exception("Observation failed", duration_ms=duration_ms)

                    if lf_obs:
                        try:
                            lf_obs.update(
                                output={"error": str(e)},
                                metadata={"duration_ms": duration_ms},
                            )
                        except Exception:
                            pass

                    if ls_run and langsmith:
                        try:
                            langsmith.update_run(
                                ls_run["id"],
                                error=str(e),
                                metadata={"duration_ms": duration_ms},
                            )
                        except Exception:
                            pass

                    raise

            return _attach_observability(result, args, kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else func

    return decorator
