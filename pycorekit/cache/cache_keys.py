"""
Stable SHA256 cache key generator for final-answer caching.
"""

import hashlib
import json
import re

def normalize_question(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)

def make_query_cache_key(thread_id: str, question: str) -> str:
    payload = json.dumps(
        {"thread_id": thread_id, "question": normalize_question(question)},
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"final_answer:{digest}"
