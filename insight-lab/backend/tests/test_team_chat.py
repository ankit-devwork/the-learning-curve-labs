"""Team chat message validation tests."""

import pytest
from pycorekit.exceptions.file import FileException

from app.services.team_chat_validation import validate_team_chat_body


def test_validate_accepts_plain_english():
    assert validate_team_chat_body("Hello team, see you at study group.") == (
        "Hello team, see you at study group."
    )


def test_validate_rejects_empty():
    with pytest.raises(FileException):
        validate_team_chat_body("   ")


def test_validate_rejects_emoji():
    with pytest.raises(FileException):
        validate_team_chat_body("Great work everyone 👍")


def test_validate_rejects_urls():
    with pytest.raises(FileException):
        validate_team_chat_body("Check https://example.com for notes")


def test_validate_rejects_file_extensions():
    with pytest.raises(FileException):
        validate_team_chat_body("I uploaded notes.pdf")


def test_validate_rejects_html():
    with pytest.raises(FileException):
        validate_team_chat_body("<script>alert(1)</script>")


def test_validate_rejects_non_ascii():
    with pytest.raises(FileException):
        validate_team_chat_body("Caf\u00e9 meetup at noon")


def test_validate_collapses_whitespace():
    assert validate_team_chat_body("  Hello   world  ") == "Hello world"
