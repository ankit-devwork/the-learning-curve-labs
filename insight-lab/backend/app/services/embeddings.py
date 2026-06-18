import os
from functools import lru_cache

import litellm

from app.core.config import settings
from app.core.exceptions import ServiceUnavailableException
from app.core.yaml_config import get_yaml_config
from app.core.migration_guard import is_missing_phase2_schema, run_or_raise_phase2


def vector_to_pgvector(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def _as_float_list(values) -> list[float]:
    """Convert numpy/onnx float32 vectors to JSON-serializable Python floats."""
    return [float(value) for value in values]


@lru_cache(maxsize=1)
def _get_fastembed_model():
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ServiceUnavailableException(
            "fastembed is not installed. Run: pip install fastembed"
        ) from exc

    model_name = get_yaml_config().embeddings.model
    return TextEmbedding(model_name=model_name)


def _embed_with_fastembed(texts: list[str]) -> list[list[float]]:
    model = _get_fastembed_model()
    return [_as_float_list(vector) for vector in model.embed(texts)]


def _embed_with_litellm(texts: list[str]) -> list[list[float]]:
    cfg = get_yaml_config().embeddings
    api_key = os.getenv("OPENAI_API_KEY") or settings.groq_api_key or None
    response = litellm.embedding(model=cfg.model, input=texts, api_key=api_key)
    embeddings = [item["embedding"] for item in response.data]
    if embeddings and len(embeddings[0]) != cfg.dimensions:
        raise ServiceUnavailableException(
            f"Embedding dimensions mismatch: model returned {len(embeddings[0])}, "
            f"expected {cfg.dimensions}. Update migration 003 or config.yaml."
        )
    return embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    cfg = get_yaml_config().embeddings
    if cfg.provider.lower() == "litellm":
        return _embed_with_litellm(texts)
    return _embed_with_fastembed(texts)


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts_batched(texts: list[str]) -> list[list[float]]:
    cfg = get_yaml_config().embeddings
    batch_size = max(cfg.batch_size, 1)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(embed_texts(batch))
    return vectors


def search_similar_chunks(
    client,
    *,
    document_id: str,
    query_embedding: list[float],
    limit: int,
    threshold: float,
) -> list[dict]:
    result = (
        client.rpc(
            "match_document_chunks",
            {
                "filter_document_id": document_id,
                "query_embedding": vector_to_pgvector(query_embedding),
                "match_count": limit,
            },
        ).execute()
    )
    rows = result.data or []
    return [row for row in rows if float(row.get("similarity", 0)) >= threshold]


def search_workspace_chunks(
    client,
    *,
    document_ids: list[str],
    query_embedding: list[float],
    limit: int,
    threshold: float,
) -> list[dict]:
    if not document_ids:
        return []
    try:
        result = (
            client.rpc(
                "match_workspace_chunks",
                {
                    "filter_document_ids": document_ids,
                    "query_embedding": vector_to_pgvector(query_embedding),
                    "match_count": limit,
                },
            ).execute()
        )
    except Exception as exc:
        if is_missing_phase2_schema(exc):
            return []
        raise
    rows = result.data or []
    return [row for row in rows if float(row.get("similarity", 0)) >= threshold]
