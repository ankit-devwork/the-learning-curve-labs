import pytest

from app.core.migration_guard import is_missing_phase2_schema


def test_is_missing_phase2_schema_detects_document_concepts_error():
    exc = Exception(
        "{'message': \"Could not find the table 'public.document_concepts' in the schema cache\", "
        "'code': 'PGRST205'}"
    )
    assert is_missing_phase2_schema(exc) is True


def test_is_missing_phase2_schema_ignores_unrelated_errors():
    assert is_missing_phase2_schema(Exception("connection timeout")) is False


def test_run_or_none_phase2_returns_none_when_schema_missing():
    from app.core.migration_guard import run_or_none_phase2

    def _failing_call():
        raise Exception("Could not find the table 'public.document_concepts' in the schema cache")

    assert run_or_none_phase2(_failing_call) is None
