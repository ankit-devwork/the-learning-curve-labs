"""Migration guard helpers for Phase 3 schema."""

from app.core.migration_guard import (
    is_missing_study_sessions_schema,
    is_missing_team_chat_schema,
)


def test_study_sessions_schema_markers():
    assert is_missing_study_sessions_schema(Exception("Could not find the table 'study_sessions'"))
    assert is_missing_study_sessions_schema(Exception("PGRST205 study_session_steps"))
    assert not is_missing_study_sessions_schema(Exception("connection timeout"))


def test_team_chat_schema_markers():
    assert is_missing_team_chat_schema(Exception("Could not find the table 'workspace_messages'"))
    assert not is_missing_team_chat_schema(Exception("rate limit exceeded"))
