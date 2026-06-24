"""Validate workspace team chat messages: plain English text only."""

from __future__ import annotations

import re

from pycorekit.exceptions.file import FileException

_ALLOWED_BODY = re.compile(r"^[A-Za-z0-9\s.,?!'\"():;\-\n\r]+$")
_BLOCKED_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\bwww\.", re.IGNORECASE),
    re.compile(r"<[^>]+>"),
    re.compile(r"\[[^\]]*\]\([^)]*\)"),
    re.compile(r"\.(pdf|docx?|xlsx?|csv|png|jpe?g|gif|webp|zip|mp3|mp4)\b", re.IGNORECASE),
)


def validate_team_chat_body(body: str, *, max_length: int = 2000) -> str:
    """Normalize and validate a team chat message body."""
    if not isinstance(body, str):
        raise FileException("Message must be text", status_code=400)

    normalized = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise FileException("Message cannot be empty", status_code=400)
    if len(normalized) > max_length:
        raise FileException(f"Message is too long (max {max_length} characters)", status_code=400)

    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(normalized):
            raise FileException(
                "Messages must be plain English text only (no links, files, HTML, or emoji)",
                status_code=400,
            )

    if not _ALLOWED_BODY.fullmatch(normalized):
        raise FileException(
            "Messages must use plain English letters, numbers, and basic punctuation only",
            status_code=400,
        )

    collapsed_lines = "\n".join(line.strip() for line in normalized.split("\n"))
    collapsed = re.sub(r"[ \t]+", " ", collapsed_lines).strip()
    if not collapsed:
        raise FileException("Message cannot be empty", status_code=400)
    return collapsed
