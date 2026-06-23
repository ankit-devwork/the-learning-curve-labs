"""Semantic similarity cache for paraphrased questions."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from app.core.cache import cache_get, cache_set
from app.core.yaml_config import get_yaml_config
from app.services.embeddings import embed_text
from app.services.llm_client import semantic_chat_index_key


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


async def get_semantic_cached_answer(
    *,
    user_id: str,
    document_id: str,
    question: str,
) -> dict[str, Any] | None:
    cfg = get_yaml_config().cache
    index_key = semantic_chat_index_key(user_id, document_id)
    index = await cache_get(index_key)
    if not index or not isinstance(index, list):
        return None

    query_embedding = embed_text(question.strip())
    best_match: dict[str, Any] | None = None
    best_score = 0.0

    for entry in index:
        embedding = entry.get("embedding")
        if not isinstance(embedding, list):
            continue
        score = _cosine_similarity(query_embedding, embedding)
        if score >= cfg.semantic_threshold and score > best_score:
            best_score = score
            best_match = entry

    if not best_match or not best_match.get("payload"):
        return None

    payload = dict(best_match["payload"])
    payload["cached"] = True
    payload["cache_match"] = "semantic"
    payload["similarity"] = round(best_score, 4)
    return payload


async def store_semantic_cached_answer(
    *,
    user_id: str,
    document_id: str,
    question: str,
    payload: dict[str, Any],
) -> None:
    cfg = get_yaml_config().cache
    index_key = semantic_chat_index_key(user_id, document_id)
    index = await cache_get(index_key)
    entries: list[dict[str, Any]] = list(index) if isinstance(index, list) else []

    embedding = embed_text(question.strip())
    entries.append(
        {
            "question": question.strip(),
            "embedding": embedding,
            "payload": payload,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    max_entries = cfg.semantic_max_entries
    if len(entries) > max_entries:
        entries = entries[-max_entries:]
    await cache_set(index_key, entries, cfg.chat_ttl)
