import pytest

from app.services.quiz_questions import draft_to_rows, parse_quiz_draft


def test_parse_quiz_draft_json():
    raw = """
    {
      "title": "Intro quiz",
      "questions": [
        {
          "question_text": "What is RAG?",
          "options": ["Retrieval Augmented Generation", "Random Access Graph", "Runtime API Gateway"],
          "correct_option_index": 0,
          "explanation": "RAG combines retrieval with generation.",
          "source_chunk_index": 0
        }
      ]
    }
    """
    draft = parse_quiz_draft(raw, max_questions=5)
    assert draft.title == "Intro quiz"
    assert len(draft.questions) == 1
    assert draft.questions[0].options[0] == "Retrieval Augmented Generation"


def test_parse_quiz_draft_rejects_invalid_correct_index():
    raw = """
    {
      "title": "Bad quiz",
      "questions": [
        {
          "question_text": "Pick one",
          "options": ["A", "B"],
          "correct_option_index": 3,
          "explanation": "Nope"
        }
      ]
    }
    """
    with pytest.raises(ValueError):
        parse_quiz_draft(raw, max_questions=5)


def test_draft_to_rows_maps_chunk_index():
    draft = parse_quiz_draft(
        """
        {
          "title": "Chunk quiz",
          "questions": [
            {
              "question_text": "Question?",
              "options": ["A", "B"],
              "correct_option_index": 1,
              "source_chunk_index": 1
            }
          ]
        }
        """,
        max_questions=5,
    )
    rows = draft_to_rows(draft, chunk_indexes=[10, 20, 30])
    assert rows[0]["source_chunk_id"] == "20"
    assert rows[0]["sort_order"] == 0
