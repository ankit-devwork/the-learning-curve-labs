import hashlib
import json
import re

def normalize_question(text: str) -> str:
    """
    Normalize question text so identical questions always map to the same cache key.
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text

def make_query_cache_key(thread_id: str, question: str) -> str:
    """
    Create a stable, deterministic cache key for final answers.
    """
    payload = json.dumps(
        {
            "thread_id": thread_id,
            "question": normalize_question(question),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"final_answer:{digest}"
