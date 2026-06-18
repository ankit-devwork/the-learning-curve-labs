"""Detect missing Phase 2 Supabase schema and return actionable errors."""

PHASE2_MIGRATION_HINT = (
    "Phase 2 database migration required. "
    "Run supabase/migrations/007_phase2_graph_mastery_multi_doc.sql in the Supabase SQL Editor, "
    "then reload the API schema (Project Settings → API → Reload)."
)

PHASE2_MIGRATION_NOTICE = PHASE2_MIGRATION_HINT


def is_missing_phase2_schema(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = (
        "document_concepts",
        "concept_relationships",
        "concept_mastery",
        "match_workspace_chunks",
        "pgrst205",
        "could not find the table",
        "schema cache",
    )
    return any(marker in message for marker in markers)


def raise_phase2_migration_required() -> None:
    from pycorekit.exceptions.file import FileException

    raise FileException(PHASE2_MIGRATION_HINT, status_code=503)


def run_or_raise_phase2(callable_fn):
    try:
        return callable_fn()
    except Exception as exc:
        if is_missing_phase2_schema(exc):
            raise_phase2_migration_required()
        raise


def run_or_none_phase2(callable_fn):
    """Like run_or_raise_phase2 but returns None when Phase 2 tables are missing."""
    try:
        return callable_fn()
    except Exception as exc:
        if is_missing_phase2_schema(exc):
            return None
        raise
