from app.services.chunking import chunk_text


def test_chunk_text_splits_long_document():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
