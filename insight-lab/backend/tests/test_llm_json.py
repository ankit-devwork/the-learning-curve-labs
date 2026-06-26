"""Tests for tolerant LLM JSON parsing."""

from app.services.llm_json import (
    escape_control_chars_in_json_strings,
    parse_llm_json,
    strip_json_fence,
)


def test_strip_json_fence_removes_markdown_wrapper():
    raw = '```json\n{"explanation": "hello"}\n```'
    assert strip_json_fence(raw) == '{"explanation": "hello"}'


def test_parse_llm_json_handles_unescaped_newlines_in_strings():
    raw = """{
  "explanation": "A ratio compares two quantities of the same type.
It helps us understand how two quantities relate to each other."
}"""
    payload = parse_llm_json(raw)
    assert "A ratio compares two quantities" in payload["explanation"]
    assert "It helps us understand" in payload["explanation"]


def test_parse_llm_json_preserves_properly_escaped_strings():
    raw = '{"explanation": "Line one\\nLine two"}'
    payload = parse_llm_json(raw)
    assert payload["explanation"] == "Line one\nLine two"


def test_escape_control_chars_in_json_strings_only_inside_quotes():
    text = '{\n  "explanation": "first\nsecond"\n}'
    repaired = escape_control_chars_in_json_strings(text)
    assert repaired == '{\n  "explanation": "first\\nsecond"\n}'
    assert parse_llm_json(text)["explanation"] == "first\nsecond"
