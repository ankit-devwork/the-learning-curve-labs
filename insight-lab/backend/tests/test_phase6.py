import math

import pytest

from app.services.export_utils import flashcards_to_anki_csv, study_guide_to_markdown


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def test_flashcards_to_anki_csv():
    csv_text = flashcards_to_anki_csv(
        [
            {"front": 'What is "RAG"?', "back": "Retrieval augmented generation"},
            {"front": "Second", "back": "Answer"},
        ],
        tag="Biology101",
    )
    assert csv_text.startswith("Front,Back,Tags")
    assert '""RAG""' in csv_text or '"What is ""RAG""?"' in csv_text
    assert "Biology101" in csv_text


def test_study_guide_to_markdown():
    markdown = study_guide_to_markdown(
        title="Chapter 1",
        content={
            "overview": "Intro to cells.",
            "key_terms": [{"term": "Mitosis", "definition": "Cell division"}],
            "sections": [{"heading": "Core ideas", "bullets": ["Cells divide", "DNA replicates"]}],
            "sample_questions": ["What is mitosis?"],
        },
    )
    assert "# Chapter 1" in markdown
    assert "## Overview" in markdown
    assert "**Mitosis**" in markdown
    assert "## Sample questions" in markdown


def test_cosine_similarity_identical():
    left = [1.0, 0.0, 0.0]
    right = [1.0, 0.0, 0.0]
    assert _cosine_similarity(left, right) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    left = [1.0, 0.0]
    right = [0.0, 1.0]
    assert _cosine_similarity(left, right) == pytest.approx(0.0)


def test_semantic_index_key_patterns():
    user_id = "u1"
    document_id = "d1"
    docs_hash = "abc"
    file_hash = "hash"
    assert f"semantic_chat:{user_id}:{document_id}" == "semantic_chat:u1:d1"
    assert f"semantic_multi_chat:{user_id}:{docs_hash}" == "semantic_multi_chat:u1:abc"
    assert f"semantic_excel_chat:{user_id}:{document_id}:{file_hash}" == "semantic_excel_chat:u1:d1:hash"
