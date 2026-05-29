import time
import asyncio
from functools import wraps
from typing import Callable, Any, Optional

from litellm import completion_cost

from app.observability.logger import logger, get_current_correlation_id
from app.observability.tracing import start_trace


def with_observability(
    name: Optional[str] = None,
    include_args: bool = False,
    include_result: bool = False,
    model: Optional[str] = None,
):
    def decorator(func: Callable):
        func_name = name or func.__name__

        def process_telemetry(result: Any, args: tuple, kwargs: dict, start_time: float, bound_logger) -> dict:
            duration = round((time.time() - start_time) * 1000, 2)
            payload = {"duration_ms": duration}

            if model and result:
                try:
                    cost_model = model
                    
                    # 🚀 DYNAMIC RESOLUTION: If provider prefix is missing (no slash), add it
                    if "/" not in cost_model:
                        # 1. First choice: Check if your property file has an explicit provider key
                        #    (e.g., settings.llm_provider, settings.provider, etc.)
                        from app.core.load_property import settings
                        provider = getattr(settings, "llm_provider", None) or getattr(settings, "provider", None)
                        
                        # 2. Fallback: If no provider property exists, check the object type
                        if not provider:
                            if hasattr(result, "x_groq") or "x_groq" in getattr(result, "__dict__", {}):
                                provider = "groq"
                            else:
                                provider = "openai" # Default safe fallback
                        
                        cost_model = f"{provider.lower()}/{cost_model}"

                    # Calculate cost using the cleanly constructed string (e.g. "groq/llama-3.3-70b-versatile")
                    cost = float(completion_cost(result, model=cost_model) or 0.0)
                    payload["cost"] = cost
                    payload["model"] = model

                    prompt = kwargs.get("prompt") or (args[0] if args else "Unknown Prompt")
                    trace = kwargs.get("__trace_obj__", {})
                    
                    ls_span = trace.get("langsmith")
                    if ls_span and hasattr(ls_span, "log_input") and hasattr(ls_span, "log_output"):
                        ls_span.log_input(str(prompt))
                        ls_span.log_output(str(result))

                    lf_span = trace.get("langfuse")
                    if lf_span:
                        lf_span.update(input=str(prompt), output=str(result))
                        lf_span.score(name="cost", value=cost, data_type="NUMERIC")

                except Exception as telemetry_err:
                    bound_logger.warning(f"Telemetry tracking bypass: {telemetry_err}")

            if include_result:
                payload["result"] = str(result)

            bound_logger.info("Span completed", **payload)
            return result

        # --- ASYNC WRAPPER ---
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            cid = get_current_correlation_id()
            bound = logger.bind(correlation_id=cid, span=func_name)

            if include_args:
                bound.info("Span started", args=str(args), kwargs=str(kwargs))
            else:
                bound.info("Span started")

            with start_trace(func_name, cid) as trace:
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    kwargs["__trace_obj__"] = trace
                    return process_telemetry(result, args, kwargs, start, bound)
                except Exception as err:
                    bound.exception("Span failed")
                    raise err

        # --- SYNC WRAPPER ---
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            cid = get_current_correlation_id()
            bound = logger.bind(correlation_id=cid, span=func_name)

            if include_args:
                bound.info("Span started", args=str(args), kwargs=str(kwargs))
            else:
                bound.info("Span started")

            with start_trace(func_name, cid) as trace:
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    kwargs["__trace_obj__"] = trace
                    return process_telemetry(result, args, kwargs, start, bound)
                except Exception as err:
                    bound.exception("Span failed")
                    raise err

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator