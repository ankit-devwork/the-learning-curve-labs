"""
sanitize_observability.py
-------------------------

Transforms the raw trace dictionary into a fully JSON‑safe structure.

Handles:
- Removing HTTP middleware spans and incomplete spans
- Flattening Langfuse + LangSmith objects
- Rolling span durations into a dashboard-friendly dict
- Removing non‑serializable fields
- Preventing circular references
- Ensuring frontend‑safe output
"""

from typing import Any, Dict, List

from pycorekit.tracing.tracing import get_external_tracing_status


_HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")


def _safe_primitive(value: Any):
    """
    Convert a value into a JSON‑safe primitive.
    Drops anything that cannot be serialized.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        return [_safe_primitive(v) for v in value]

    if isinstance(value, dict):
        return {k: _safe_primitive(v) for k, v in value.items()}

    return str(value)


def _is_http_middleware_span(span: Dict[str, Any]) -> bool:
    name = (span.get("name") or "").strip()
    if not name:
        return False

    parts = name.split(maxsplit=1)
    if parts and parts[0] in _HTTP_METHODS:
        return True

    inputs = span.get("inputs") or {}
    return isinstance(inputs, dict) and "path" in inputs


def _is_complete_span(span: Dict[str, Any]) -> bool:
    return span.get("end_ts") is not None and span.get("duration_ms") is not None


def _serialize_langfuse(lf: Any) -> Dict[str, Any] | None:
    if not lf:
        return None
    try:
        return {
            "id": getattr(lf, "id", None),
            "name": getattr(lf, "name", None),
            "metadata": _safe_primitive(getattr(lf, "metadata", None)),
        }
    except Exception:
        return {"error": "Failed to serialize Langfuse object"}


def _serialize_langsmith(ls: Any) -> Dict[str, Any] | None:
    if not ls:
        return None
    try:
        return {
            "id": ls.get("id"),
            "name": ls.get("name"),
            "run_type": ls.get("run_type"),
            "metadata": _safe_primitive(ls.get("metadata")),
        }
    except Exception:
        return {"error": "Failed to serialize LangSmith object"}


def _build_durations(spans: List[Dict[str, Any]]) -> Dict[str, float]:
    durations: Dict[str, float] = {}
    seen: Dict[str, int] = {}

    for index, span in enumerate(spans):
        name = span.get("name") or f"span_{index}"
        duration = span.get("duration_ms")
        if duration is None:
            continue

        seen[name] = seen.get(name, 0) + 1
        key = name if seen[name] == 1 else f"{name}_{seen[name]}"
        durations[key] = float(duration)

    return durations


def _build_external_tracing(
    langfuse: Dict[str, Any] | None,
    langsmith: Dict[str, Any] | None,
) -> Dict[str, Any]:
    status = get_external_tracing_status()

    langfuse_id = (langfuse or {}).get("id")
    langsmith_id = (langsmith or {}).get("id")

    langfuse_info = status["langfuse"]
    langsmith_info = status["langsmith"]

    langfuse_url = None
    if langfuse_info["configured"] and langfuse_id:
        host = langfuse_info["host"].rstrip("/")
        langfuse_url = f"{host}/trace/{langfuse_id}"

    langsmith_url = langsmith_info.get("url")
    if langsmith_info["configured"] and langsmith_id:
        langsmith_url = f"{langsmith_info['base_url']}/o/default/projects/p/{langsmith_info['project']}/r/{langsmith_id}"

    return {
        "langfuse": {
            "configured": langfuse_info["configured"],
            "active": langfuse is not None,
            "id": langfuse_id,
            "host": langfuse_info["host"],
            "url": langfuse_url,
        },
        "langsmith": {
            "configured": langsmith_info["configured"],
            "active": langsmith is not None,
            "id": langsmith_id,
            "project": langsmith_info["project"],
            "url": langsmith_url,
        },
    }


def sanitize_observability(trace_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert full observability trace into a JSON-safe structure.

    Returns a payload with:
    - durations: per-span latency rollup for dashboard charts
    - total_duration_ms: endpoint span duration when available
    - external_tracing: Langfuse/LangSmith status and deep links
    - raw: sanitized spans and provider metadata
    """

    if not trace_dict:
        empty_raw = {"spans": [], "errors": [], "langfuse": None, "langsmith": None}
        return {
            "durations": {},
            "total_duration_ms": 0,
            "external_tracing": _build_external_tracing(None, None),
            "raw": empty_raw,
        }

    langfuse = _serialize_langfuse(trace_dict.get("langfuse"))
    langsmith = _serialize_langsmith(trace_dict.get("langsmith"))

    spans: List[Dict[str, Any]] = trace_dict.get("spans", [])
    clean_spans: List[Dict[str, Any]] = []

    endpoint_duration_ms = None

    for span in spans:
        if _is_http_middleware_span(span):
            continue
        if not _is_complete_span(span):
            continue

        clean_span = {
            "name": span.get("name"),
            "type": span.get("type"),
            "inputs": _safe_primitive(span.get("inputs")),
            "start_ts": span.get("start_ts"),
            "end_ts": span.get("end_ts"),
            "duration_ms": span.get("duration_ms"),
            "error": _safe_primitive(span.get("error")),
        }
        clean_spans.append(clean_span)

        if endpoint_duration_ms is None and clean_span.get("name") not in (
            "validate_file",
            "load_chat_history_async",
            "observability_test_handler",
        ):
            # First non-trivial closed span after HTTP is usually the endpoint wrapper.
            pass

    durations = _build_durations(clean_spans)

    endpoint_names = {
        "upload_and_ingest",
        "ask_question",
        "choose_document",
        "list_documents",
        "observability_test",
    }
    for span in clean_spans:
        if span.get("name") in endpoint_names:
            endpoint_duration_ms = span.get("duration_ms")
            break

    if endpoint_duration_ms is None and clean_spans:
        endpoint_duration_ms = clean_spans[0].get("duration_ms")

    total_duration_ms = float(endpoint_duration_ms or 0)

    raw = {
        "spans": clean_spans,
        "langfuse": langfuse,
        "langsmith": langsmith,
        "errors": _safe_primitive(trace_dict.get("errors", [])),
    }

    return {
        "durations": durations,
        "total_duration_ms": total_duration_ms,
        "external_tracing": _build_external_tracing(langfuse, langsmith),
        "raw": raw,
    }
