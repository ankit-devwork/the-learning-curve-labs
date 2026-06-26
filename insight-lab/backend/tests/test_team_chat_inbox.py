"""Team chat inbox and read-state helpers."""

from pathlib import Path

SERVICE_PATH = Path(__file__).resolve().parents[1] / "app/services/workspace_messages_service.py"


def test_workspace_messages_service_exports_inbox_helpers():
    text = SERVICE_PATH.read_text(encoding="utf-8")
    assert "async def list_chat_inbox" in text
    assert "async def mark_workspace_messages_read" in text
    assert "def _attach_read_receipts" in text
