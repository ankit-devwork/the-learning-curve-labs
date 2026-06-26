"""Excel chat history source serialization."""

from app.services.chat_history_format import format_excel_chat_sources, parse_excel_chat_sources


def test_format_and_parse_excel_chat_sources():
    stored = format_excel_chat_sources(
        ["Revenue", "Month"],
        [{"document_id": "doc-1", "filename": "notes.pdf", "excerpt": "Ratio basics"}],
    )
    columns, citations = parse_excel_chat_sources(stored)
    assert columns == ["Revenue", "Month"]
    assert len(citations) == 1
    assert citations[0]["filename"] == "notes.pdf"
