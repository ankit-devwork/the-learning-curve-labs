"""Detect missing Supabase schema migrations and return actionable errors."""

PHASE2_MIGRATION_HINT = (
    "Phase 2 database migration required. "
    "Run supabase/migrations/007_phase2_graph_mastery_multi_doc.sql in the Supabase SQL Editor, "
    "then reload the API schema (Project Settings → API → Reload)."
)

PHASE2_MIGRATION_NOTICE = PHASE2_MIGRATION_HINT

PHASE3_016_MIGRATION_HINT = (
    "Study session migration required. "
    "Run supabase/migrations/016_study_sessions_learning_paths.sql in the Supabase SQL Editor, "
    "then reload the API schema (Project Settings → API → Reload)."
)

PHASE3_016_MIGRATION_NOTICE = PHASE3_016_MIGRATION_HINT

PHASE3_017_MIGRATION_HINT = (
    "Team chat migration required. "
    "Run supabase/migrations/017_workspace_team_chat.sql in the Supabase SQL Editor, "
    "then reload the API schema (Project Settings → API → Reload)."
)

PHASE3_017_MIGRATION_NOTICE = PHASE3_017_MIGRATION_HINT


def _message_has_markers(message: str, markers: tuple[str, ...]) -> bool:
    return any(marker in message for marker in markers)


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
    return _message_has_markers(message, markers)


def is_missing_study_sessions_schema(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = (
        "study_sessions",
        "study_session_steps",
        "learning_paths",
        "learning_path_nodes",
        "pgrst205",
        "could not find the table",
        "schema cache",
    )
    return _message_has_markers(message, markers)


def is_missing_team_chat_schema(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = (
        "workspace_messages",
        "pgrst205",
        "could not find the table",
        "schema cache",
    )
    return _message_has_markers(message, markers)


def raise_phase2_migration_required() -> None:
    from pycorekit.exceptions.file import FileException

    raise FileException(PHASE2_MIGRATION_HINT, status_code=503)


def raise_study_sessions_migration_required() -> None:
    from pycorekit.exceptions.file import FileException

    raise FileException(PHASE3_016_MIGRATION_HINT, status_code=503)


def raise_team_chat_migration_required() -> None:
    from pycorekit.exceptions.file import FileException

    raise FileException(PHASE3_017_MIGRATION_HINT, status_code=503)


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


def run_or_raise_study_sessions(callable_fn):
    try:
        return callable_fn()
    except Exception as exc:
        if is_missing_study_sessions_schema(exc):
            raise_study_sessions_migration_required()
        raise


def run_or_none_study_sessions(callable_fn):
    try:
        return callable_fn()
    except Exception as exc:
        if is_missing_study_sessions_schema(exc):
            return None
        raise


def run_or_raise_team_chat(callable_fn):
    try:
        return callable_fn()
    except Exception as exc:
        if is_missing_team_chat_schema(exc):
            raise_team_chat_migration_required()
        raise


def run_or_none_team_chat(callable_fn):
    try:
        return callable_fn()
    except Exception as exc:
        if is_missing_team_chat_schema(exc):
            return None
        raise
