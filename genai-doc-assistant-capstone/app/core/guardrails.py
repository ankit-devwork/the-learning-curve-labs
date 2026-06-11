import re
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.service.db_connection import async_embed_texts

UNSAFE_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bdelete\s+all\b",
    r"\bdrop\s+table\b",
    r"\bformat\s+disk\b",
    r"\bshutdown\b",
    r"\berase\s+everything\b",
    r"\bwipe\s+data\b",
    r"\bremove\s+all\b",
    r"\bkill\s+process\b",
    r"\bdisable\s+security\b",
    r"\bmodify\s+system32\b",
]

UNSAFE_CONTENT_PATTERNS = [
    r"\bhow\s+to\s+hack\b",
    r"\bexploit\b",
    r"\bzero[-\s]?day\b",
    r"\bmake\s+a\s+bomb\b",
    r"\bbuild\s+an\s+explosive\b",
    r"\bself[-\s]?harm\b",
    r"\bsuicide\b",
    r"\bkill\s+myself\b",
    r"\bkill\s+someone\b",
]


def detect_unsafe_ops(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in UNSAFE_PATTERNS)


def detect_unsafe_content(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in UNSAFE_CONTENT_PATTERNS)


def apply_guardrails(answer: str) -> str:
    if detect_unsafe_ops(answer) or detect_unsafe_content(answer):
        return "I cannot assist with unsafe, destructive, or harmful actions."
    return answer


def _split_context_chunks(context: str, retrieved_chunks: List[str] | None = None) -> List[str]:
    if retrieved_chunks:
        chunks = [c.strip() for c in retrieved_chunks if c and str(c).strip()]
        if chunks:
            return chunks
    if not context or not context.strip():
        return []
    return [part.strip() for part in context.split("\n\n") if part.strip()]


async def detect_hallucination_async(
    answer: str,
    context: str,
    retrieved_chunks: List[str] | None = None,
    threshold: float = 0.55,
) -> bool:
    """
    Chunk-level grounding check.

    Embeds the answer and each context chunk, then uses the maximum similarity.
    This is more reliable than comparing the answer to one giant context vector.
    """
    if not answer.strip():
        return True

    chunks = _split_context_chunks(context, retrieved_chunks)
    if not chunks:
        return True

    texts = [answer.strip(), *chunks]
    embeddings = await async_embed_texts(texts)
    answer_emb = np.array(embeddings[0]).reshape(1, -1)
    chunk_embs = np.array(embeddings[1:])

    similarities = cosine_similarity(answer_emb, chunk_embs)[0]
    max_sim = float(np.max(similarities))
    return max_sim < threshold
