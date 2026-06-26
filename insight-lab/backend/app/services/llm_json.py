"""Parse JSON returned by LLMs, tolerating common formatting mistakes."""

from __future__ import annotations

import json
from typing import Any


def strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def escape_control_chars_in_json_strings(text: str) -> str:
    """Escape raw newlines/tabs inside JSON string literals."""
    out: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ord(ch) < 32:
            if ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(f"\\u{ord(ch):04x}")
            continue
        out.append(ch)

    return "".join(out)


def parse_llm_json(raw: str) -> Any:
    text = strip_json_fence(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(escape_control_chars_in_json_strings(text))
