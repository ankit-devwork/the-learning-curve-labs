from pycorekit.utils.env_overrides import apply_env_overrides


def test_nested_env_override(monkeypatch):
    monkeypatch.setenv("APP_RAG__CHUNK_SIZE", "999")
    data = {"rag": {"chunk_size": 300, "chunk_overlap": 50}}
    result = apply_env_overrides(data, prefix="APP")
    assert result["rag"]["chunk_size"] == 999
    assert result["rag"]["chunk_overlap"] == 50


def test_top_level_env_override(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    data = {"env": "dev"}
    result = apply_env_overrides(data, prefix="APP")
    assert result["env"] == "prod"
