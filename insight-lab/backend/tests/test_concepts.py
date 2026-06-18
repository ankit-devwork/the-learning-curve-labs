import pytest

from app.services.concept_extraction import (
    normalize_concept_id,
    parse_concept_extraction,
    draft_to_rows,
)


def test_normalize_concept_id():
    assert normalize_concept_id("Machine Learning") == "machine-learning"


def test_parse_concept_extraction():
    raw = """
    {
      "concepts": [
        {"id": "rag", "name": "RAG", "topic": "AI", "chunk_indexes": [0, 1]},
        {"name": "Embeddings", "chunk_indexes": [2]}
      ],
      "relationships": [
        {"source_id": "embeddings", "target_id": "rag", "type": "prerequisite_for"}
      ]
    }
    """
    draft = parse_concept_extraction(raw, max_concepts=10)
    assert len(draft.concepts) == 2
    concept_rows, rel_rows = draft_to_rows(draft, document_id="doc-1")
    assert concept_rows[0]["concept_id"] == "rag"
    assert rel_rows[0]["relationship_type"] == "prerequisite_for"


def test_parse_concept_extraction_requires_concepts():
    with pytest.raises(ValueError):
        parse_concept_extraction('{"concepts": []}', max_concepts=5)
