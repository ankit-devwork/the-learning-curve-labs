import time
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
    """
    Full non-intrusive observability decorator:
    - Auto reads ContextVar for structured traces
    - Manages LangSmith + Langfuse spans cleanly
    - Dynamically evaluates downstream LiteLLM outputs & metrics
    """
    def decorator(func: Callable):
        func_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Silently inherit context via thread/coroutine token
            cid = get_current_correlation_id()
            bound = logger.bind(correlation_id=cid, span=func_name)

            if include_args:
                bound.info("Span started", args=str(args), kwargs=str(kwargs))
            else:
                bound.info("Span started")

            with start_trace(func_name, cid) as trace:
                start = time.time()

                try:
                    # Execute actual function execution path smoothly
                    result = func(*args, **kwargs)
                    
                    duration = round((time.time() - start) * 1000, 2)
                    payload = {"duration_ms": duration}

                    # Post-processing hook if tracking an active AI model signature
                    if model and result:
                        try:
                            # Parse structural cost tracking calculations from output object
                            cost = float(completion_cost(result) or 0.0)
                            payload["cost"] = cost
                            payload["model"] = model

                            prompt = kwargs.get("prompt") or (args[0] if args else "Unknown Prompt")

                            # Push metrics to LangSmith safely
                            ls_span = trace.get("langsmith")
                            if ls_span and hasattr(ls_span, "log_input") and hasattr(ls_span, "log_output"):
                                ls_span.log_input(str(prompt))
                                ls_span.log_output(str(result))

                            # Push metrics to Langfuse safely
                            lf_span = trace.get("langfuse")
                            if lf_span:
                                lf_span.update(input=str(prompt), output=str(result))
                                lf_span.score(name="cost", value=cost, data_type="NUMERIC")

                        except Exception as telemetry_err:
                            bound.warning(f"Telemetry tracking bypass: {telemetry_err}")

                    if include_result:
                        payload["result"] = str(result)

                    bound.info("Span completed", **payload)
                    return result

                except Exception as err:
                    bound.exception("Span failed")
                    raise err

        return wrapper
    return decorator