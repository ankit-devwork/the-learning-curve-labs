import pytest

from app.services.citations import (
    build_source_citations,
    collapse_sources_by_document,
    hash_document_ids,
    hash_source_refs,
    make_source_preview,
    strip_excerpt_markers,
)


def test_make_source_preview_truncates():
    long_text = "word " * 100
    preview = make_source_preview(long_text, max_len=50)
    assert len(preview) <= 51
    assert preview.endswith("…")


def test_make_source_preview_prefers_sentence_boundary():
    text = (
        "First sentence is complete. Second sentence adds more detail. "
        "Third sentence continues with even more background information."
    )
    preview = make_source_preview(text, max_len=80)
    assert preview.endswith(".")
    assert "Third sentence" not in preview


def test_collapse_sources_by_document_keeps_best_match():
    sources = build_source_citations(
        [
            {
                "document_id": "doc-1",
                "filename": "a.pdf",
                "chunk_index": 0,
                "content": "Lower match chunk.",
                "similarity": 0.55,
            },
            {
                "document_id": "doc-1",
                "filename": "a.pdf",
                "chunk_index": 1,
                "content": "Higher match chunk.",
                "similarity": 0.91,
            },
            {
                "document_id": "doc-2",
                "filename": "b.pdf",
                "chunk_index": 0,
                "content": "Other document chunk.",
                "similarity": 0.7,
            },
        ]
    )
    collapsed = collapse_sources_by_document(sources)
    assert len(collapsed) == 2
    assert collapsed[0]["preview"] == "Higher match chunk."
    assert collapsed[1]["filename"] == "b.pdf"


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
