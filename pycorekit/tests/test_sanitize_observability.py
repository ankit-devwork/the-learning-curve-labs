from pycorekit.utils.sanitize_observability import sanitize_observability


def _make_span(name, duration_ms, *, span_type=None, inputs=None, complete=True):
    span = {
        "name": name,
        "type": span_type,
        "inputs": inputs or {},
        "start_ts": 1.0,
        "end_ts": None if not complete else 2.0,
        "duration_ms": None if not complete else duration_ms,
        "error": None,
    }
    return span


def test_sanitize_drops_http_and_incomplete_spans():
    trace = {
        "langfuse": None,
        "langsmith": None,
        "errors": [],
        "spans": [
            _make_span("POST /upload-and-ingest", 100, inputs={"path": "/upload-and-ingest"}),
            _make_span("upload_and_ingest", 5000, complete=False),
            _make_span("validate_file", 1.2),
            _make_span("parse_file", 250.5),
        ],
    }

    safe = sanitize_observability(trace)

    span_names = [span["name"] for span in safe["raw"]["spans"]]
    assert "POST /upload-and-ingest" not in span_names
    assert "upload_and_ingest" not in span_names
    assert span_names == ["validate_file", "parse_file"]
    assert safe["durations"]["validate_file"] == 1.2
    assert safe["durations"]["parse_file"] == 250.5


def test_sanitize_includes_closed_endpoint_span_and_total_duration():
    trace = {
        "langfuse": None,
        "langsmith": None,
        "errors": [],
        "spans": [
            _make_span("POST /upload-and-ingest", 9000, inputs={"path": "/upload-and-ingest"}),
            _make_span("upload_and_ingest", 8500),
            _make_span("validate_file", 0.7),
        ],
    }

    safe = sanitize_observability(trace)

    assert safe["total_duration_ms"] == 8500
    assert safe["durations"]["upload_and_ingest"] == 8500
    assert safe["durations"]["validate_file"] == 0.7


def test_sanitize_marks_db_spans_and_external_tracing_status():
    trace = {
        "langfuse": None,
        "langsmith": None,
        "errors": [],
        "spans": [
            _make_span("POST /ask-question", 100, inputs={"path": "/ask-question"}),
            _make_span("chroma_query", 12.5, span_type="db", inputs={"query": "collection.query"}),
        ],
    }

    safe = sanitize_observability(trace)

    db_span = safe["raw"]["spans"][0]
    assert db_span["type"] == "db"
    assert safe["external_tracing"]["langfuse"]["configured"] is False
    assert safe["external_tracing"]["langsmith"]["configured"] is False
