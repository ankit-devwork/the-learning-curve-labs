"""HTTP headers for request correlation / tracking IDs."""

from __future__ import annotations

from starlette.requests import Request

from pycorekit.correlation.context import generate_correlation_id

TRACKING_ID_HEADER = "X-Tracking-ID"
CORRELATION_ID_HEADER = "x-correlation-id"

# Prefer explicit tracking id from clients; fall back to legacy correlation header.
INCOMING_TRACKING_HEADERS = ("x-tracking-id", "x-correlation-id")


def resolve_request_correlation_id(request: Request) -> str:
    """Read tracking id from incoming headers or generate a new one."""
    for header_name in INCOMING_TRACKING_HEADERS:
        value = request.headers.get(header_name)
        if value and value.strip():
            return value.strip()
    return generate_correlation_id()


def tracking_response_headers(correlation_id: str | None) -> dict[str, str]:
    """Headers to attach to every API response for log lookup."""
    cid = (correlation_id or "").strip()
    if not cid:
        return {}
    return {
        TRACKING_ID_HEADER: cid,
        CORRELATION_ID_HEADER: cid,
    }
