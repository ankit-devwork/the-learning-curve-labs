"""
sanitize_observability.py
-------------------------

Transforms the raw trace dictionary into a fully JSON‑safe structure.

Handles:
- Removing the open top‑level HTTP span (always index 0)
- Flattening Langfuse + LangSmith objects
- Removing non‑serializable fields
- Preventing circular references
- Ensuring frontend‑safe output
"""

from typing import Any, Dict, List


def _safe_primitive(value: Any):
    """
    Convert a value into a JSON‑safe primitive.
    Drops anything that cannot be serialized.
    """
    if value is None:
        return None

    # Already JSON‑safe
    if isinstance(value, (str, int, float, bool)):
        return value

    # Lists → sanitize each element
    if isinstance(value, list):
        return [_safe_primitive(v) for v in value]

    # Dicts → sanitize each value
    if isinstance(value, dict):
        return {k: _safe_primitive(v) for k, v in value.items()}

    # Fallback: convert to string
    return str(value)


def sanitize_observability(trace_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert full observability trace into a JSON-safe structure.

    This function drops the open HTTP span, serializes non-primitive
    values, and flattens Langfuse/LangSmith objects.

    Example:
        safe_trace = sanitize_observability(request.state.trace)
    """

    if not trace_dict:
        return {"spans": [], "errors": [], "langfuse": None, "langsmith": None}

    safe: Dict[str, Any] = {}

    # ---------------------------------------------------------
    # Langfuse
    # ---------------------------------------------------------
    lf = trace_dict.get("langfuse")
    if lf:
        try:
            safe["langfuse"] = {
                "id": getattr(lf, "id", None),
                "name": getattr(lf, "name", None),
                "metadata": _safe_primitive(getattr(lf, "metadata", None)),
            }
        except Exception:
            safe["langfuse"] = {"error": "Failed to serialize Langfuse object"}
    else:
        safe["langfuse"] = None

    # ---------------------------------------------------------
    # LangSmith
    # ---------------------------------------------------------
    ls = trace_dict.get("langsmith")
    if ls:
        try:
            safe["langsmith"] = {
                "id": ls.get("id"),
                "name": ls.get("name"),
                "run_type": ls.get("run_type"),
                "metadata": _safe_primitive(ls.get("metadata")),
            }
        except Exception:
            safe["langsmith"] = {"error": "Failed to serialize LangSmith object"}
    else:
        safe["langsmith"] = None

    # ---------------------------------------------------------
    # Spans (drop top-level HTTP span)
    # ---------------------------------------------------------
    spans: List[Dict[str, Any]] = trace_dict.get("spans", [])
    clean_spans: List[Dict[str, Any]] = []

    # Skip index 0 → open HTTP span
    for span in spans[1:]:
        clean_spans.append({
            "name": span.get("name"),
            "inputs": _safe_primitive(span.get("inputs")),
            "start_ts": span.get("start_ts"),
            "end_ts": span.get("end_ts"),
            "duration_ms": span.get("duration_ms"),
            "error": _safe_primitive(span.get("error")),
        })

    safe["spans"] = clean_spans

    # ---------------------------------------------------------
    # Errors
    # ---------------------------------------------------------
    safe["errors"] = _safe_primitive(trace_dict.get("errors", []))

    return safe
