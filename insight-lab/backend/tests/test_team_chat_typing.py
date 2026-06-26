"""Team chat typing presence service checks."""

from pathlib import Path

SERVICE_PATH = Path(__file__).resolve().parents[1] / "app/services/workspace_typing_service.py"


def test_workspace_typing_service_exports():
    text = SERVICE_PATH.read_text(encoding="utf-8")
    assert "async def set_workspace_typing" in text
    assert "async def list_workspace_typing" in text
