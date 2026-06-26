"""Pure helpers for serializing Excel chat sources in history rows."""

from __future__ import annotations

from typing import Any


def format_excel_chat_sources(
    column_sources: list[str] | None,
    document_citations: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    stored: list[dict[str, Any]] = [
        {"type": "column", "label": str(column)}
        for column in (column_sources or [])
        if str(column).strip()
    ]
    stored.extend(document_citations or [])
    return stored


def parse_excel_chat_sources(
    sources: list[dict[str, Any]] | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    columns: list[str] = []
    citations: list[dict[str, Any]] = []
    for item in sources or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "column":
            label = str(item.get("label") or "").strip()
            if label:
                columns.append(label)
        else:
            citations.append(item)
    return columns, citations
