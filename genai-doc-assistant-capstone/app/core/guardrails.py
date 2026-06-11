# guardrails.py
import re
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.service.db_connection import async_embed_texts

# --------- UNSAFE OPS ---------
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

def detect_unsafe_ops(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in UNSAFE_PATTERNS)


# --------- UNSAFE CONTENT ---------
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

def detect_unsafe_content(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in UNSAFE_CONTENT_PATTERNS)


def apply_guardrails(answer: str) -> str:
    if detect_unsafe_ops(answer) or detect_unsafe_content(answer):
        return "I cannot assist with unsafe, destructive, or harmful actions."
    return answer


# --------- ASYNC HALLUCINATION DETECTOR ---------

async def detect_hallucination_async(answer: str, context: str, threshold: float = 0.75) -> bool:
    """
    Async embedding-based hallucination detection.
    Compares the answer embedding to the full concatenated context embedding.
    """

    # Empty answer → hallucination
    if not answer.strip():
        return True

    # Empty context → hallucination
    if not context or not context.strip():
        return True

    # Normalize
    answer = answer.strip()
    context = context.strip()

    # Embed both
    embeddings = await async_embed_texts([answer, context])
    ans_emb = np.array(embeddings[0]).reshape(1, -1)
    ctx_emb = np.array(embeddings[1]).reshape(1, -1)

    # Cosine similarity
    sim = float(cosine_similarity(ans_emb, ctx_emb)[0][0])

    # Hallucination if similarity too low
    return sim < threshold
