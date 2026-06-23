import pytest

from pycorekit.exceptions.file import FileException

from app.services.artifact_parsing import parse_flashcard_draft, parse_study_guide_draft
from app.services.workspace_service import _sanitize_workspace_name


def test_sanitize_workspace_name_rejects_empty():
    with pytest.raises(FileException):
        _sanitize_workspace_name("   ")


def test_sanitize_workspace_name_trims_and_collapses_spaces():
    assert _sanitize_workspace_name("  Biology   101  ") == "Biology 101"


def test_parse_flashcard_draft_limits_cards():
    raw = """
    {
      "title": "Cards",
      "cards": [
        {"front": "A", "back": "1"},
        {"front": "B", "back": "2"},
        {"front": "C", "back": "3"}
      ]
    }
    """
    draft = parse_flashcard_draft(raw, max_cards=2)
    assert len(draft.cards) == 2


def test_parse_study_guide_draft():
    raw = """
    {
      "title": "Guide",
      "overview": "Overview text",
      "key_terms": [{"term": "RAG", "definition": "Retrieval augmented generation"}],
      "sections": [{"heading": "Intro", "bullets": ["Point one"]}],
      "sample_questions": ["What is RAG?"]
    }
    """
    draft = parse_study_guide_draft(raw)
    assert draft.title == "Guide"
    assert draft.key_terms[0].term == "RAG"
