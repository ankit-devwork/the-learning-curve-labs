from starlette.requests import Request

from pycorekit.correlation.headers import (
    CORRELATION_ID_HEADER,
    TRACKING_ID_HEADER,
    resolve_request_correlation_id,
    tracking_response_headers,
)


def _request(headers: list[tuple[bytes, bytes]]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "headers": headers,
    }
    return Request(scope)


def test_resolve_prefers_x_tracking_id():
    request = _request(
        [
            (b"x-tracking-id", b"track-123"),
            (b"x-correlation-id", b"corr-456"),
        ]
    )
    assert resolve_request_correlation_id(request) == "track-123"


def test_resolve_falls_back_to_correlation_header():
    request = _request([(b"x-correlation-id", b"corr-456")])
    assert resolve_request_correlation_id(request) == "corr-456"


def test_resolve_generates_when_missing():
    request = _request([])
    generated = resolve_request_correlation_id(request)
    assert isinstance(generated, str)
    assert len(generated) >= 8


def test_tracking_response_headers():
    headers = tracking_response_headers("abc-123")
    assert headers[TRACKING_ID_HEADER] == "abc-123"
    assert headers[CORRELATION_ID_HEADER] == "abc-123"
