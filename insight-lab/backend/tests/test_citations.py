import pytest

from app.services.citations import (
    build_source_citations,
    hash_document_ids,
    hash_source_refs,
    make_source_preview,
    strip_excerpt_markers,
)


def test_make_source_preview_truncates():
    long_text = "word " * 100
    preview = make_source_preview(long_text, max_len=50)
    assert len(preview) <= 50
    assert preview.endswith("…")


def test_strip_excerpt_markers():
    answer = "RAG helps grounding. [excerpt_2] It reduces hallucinations."
    assert strip_excerpt_markers(answer) == "RAG helps grounding. It reduces hallucinations."


def test_build_source_citations_uses_filename_not_chunk_label():
    rows = [
        {
            "document_id": "doc-1",
            "filename": "lecture-notes.pdf",
            "chunk_index": 7,
            "content": "Retrieval augmented generation combines search with LLM output.",
            "similarity": 0.91,
        }
    ]
    sources = build_source_citations(rows)
    assert sources[0]["filename"] == "lecture-notes.pdf"
    assert sources[0]["chunk_index"] == 7
    assert "Retrieval augmented" in sources[0]["preview"]
    assert "chunk" not in sources[0]["preview"].lower()


def test_hash_document_ids_stable():
    assert hash_document_ids(["b", "a"]) == hash_document_ids(["a", "b"])


def test_hash_source_refs_stable():
    refs = [
        {"document_id": "b", "chunk_index": 2},
        {"document_id": "a", "chunk_index": 1},
    ]
    assert hash_source_refs(refs) == hash_source_refs(list(reversed(refs)))
