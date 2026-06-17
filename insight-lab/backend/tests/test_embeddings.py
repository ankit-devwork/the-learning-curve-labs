from app.services.embeddings import _as_float_list, vector_to_pgvector


def test_vector_to_pgvector_format():
    result = vector_to_pgvector([0.1, 0.2, 0.3])
    assert result == "[0.10000000,0.20000000,0.30000000]"


def test_as_float_list_serializes_numpy_float32():
    import json

    try:
        import numpy as np
    except ImportError:
        return

    converted = _as_float_list([np.float32(0.1), np.float32(0.2)])
    assert json.dumps(converted) == "[0.10000000149011612, 0.20000000298023224]"
    assert all(isinstance(value, float) for value in converted)
