"""Pure helpers for course pack generation."""

from __future__ import annotations

from typing import Any


def sample_homework_question(study_guide_payload: dict[str, Any] | None) -> str | None:
    if study_guide_payload:
        content = study_guide_payload.get("content") or {}
        sample_questions = content.get("sample_questions") or []
        for question in sample_questions:
            text = str(question).strip()
            if text:
                return text
        title = str(study_guide_payload.get("title") or "").strip()
        if title:
            return f"Explain the main ideas covered in {title}."
    return "Explain the key concepts from this document step by step."
