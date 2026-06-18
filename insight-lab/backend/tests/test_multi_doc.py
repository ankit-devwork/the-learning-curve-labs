from app.services.multi_doc_service import _diversify_context_rows, _retrieve_candidate_limit


def _row(document_id: str, chunk_index: int, similarity: float) -> dict:
    return {
        "document_id": document_id,
        "filename": f"{document_id}.pdf",
        "chunk_index": chunk_index,
        "content": f"chunk {chunk_index}",
        "similarity": similarity,
    }


def test_diversify_context_rows_caps_per_document():
    rows = [
        _row("doc-a", 0, 0.9),
        _row("doc-a", 1, 0.88),
        _row("doc-a", 2, 0.87),
        _row("doc-a", 3, 0.86),
        _row("doc-b", 0, 0.85),
        _row("doc-b", 1, 0.84),
    ]
    selected = _diversify_context_rows(rows, limit=6, max_per_document=3)
    assert len(selected) == 5
    assert sum(1 for row in selected if row["document_id"] == "doc-a") == 3
    assert sum(1 for row in selected if row["document_id"] == "doc-b") == 2


def test_diversify_context_rows_prefers_higher_similarity_within_cap():
    rows = [
        _row("doc-a", 0, 0.6),
        _row("doc-a", 1, 0.9),
        _row("doc-b", 0, 0.8),
    ]
    selected = _diversify_context_rows(rows, limit=2, max_per_document=1)
    assert [row["chunk_index"] for row in selected] == [1, 0]


def test_retrieve_candidate_limit_scales_with_document_count():
    class Cfg:
        max_context_chunks = 6
        max_chunks_per_document = 3

    assert _retrieve_candidate_limit(Cfg(), 2) == 12
    assert _retrieve_candidate_limit(Cfg(), 5) == 30
