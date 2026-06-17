from app.services.embeddings import vector_to_pgvector


def test_vector_to_pgvector_format():
    result = vector_to_pgvector([0.1, 0.2, 0.3])
    assert result == "[0.10000000,0.20000000,0.30000000]"
