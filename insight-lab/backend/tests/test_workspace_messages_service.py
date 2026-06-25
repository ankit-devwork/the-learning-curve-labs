"""Static checks for workspace team chat service."""

from pathlib import Path

SERVICE_PATH = Path(__file__).resolve().parents[1] / "app/services/workspace_messages_service.py"


def test_workspace_messages_service_imports_dependencies():
    """Guard against NameError 500s from missing service imports."""
    text = SERVICE_PATH.read_text(encoding="utf-8")
    assert "from app.services.workspace_access import" in text
    assert "require_workspace_role" in text
    assert "get_workspace_membership_role" in text
    assert "from app.services.team_chat_validation import validate_team_chat_body" in text
