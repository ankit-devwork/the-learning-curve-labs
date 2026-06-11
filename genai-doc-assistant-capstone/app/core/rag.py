# rag.py
import re
import hashlib
from typing import List, Callable
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Lightweight built-in stopwords (no NLTK dependency)
STOPWORDS = {
    "the","a","an","and","or","is","are","to","of","in","for","on","with","as","by",
    "this","that","it","be","from","at","which","but","not","have","has","was","were",
    "can","could","should","would","will","shall","do","does","did"
}

# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def basic_clean(text: str) -> str:
    """
    Lightweight text normalization:
    - Replace newlines/tabs with spaces
    - Collapse multiple spaces
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""

    cleaned = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Chunking Strategies
# ---------------------------------------------------------------------------

def sliding_window_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Fixed-size sliding window chunking with overlap.
    """
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == length:
            break

        start = end - chunk_overlap

    return chunks


def recursive_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Semantic-aware chunking using RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


def hybrid_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Hybrid chunking:
    - Recursive semantic chunking
    - Sliding-window fallback for oversized chunks
    """
    initial_chunks = recursive_chunk(text, chunk_size, chunk_overlap)
    final_chunks = []

    for chunk in initial_chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            sliced = sliding_window_chunk(chunk, chunk_size, chunk_overlap)
            final_chunks.extend(sliced)

    return final_chunks


def chunk_text(text: str, chunk_size: int, chunk_overlap: int, splitter: str) -> List[str]:
    """
    Unified chunking interface.

    splitter:
        - "sliding"
        - "recursive"
        - "hybrid"
    """
    if splitter == "recursive":
        return recursive_chunk(text, chunk_size, chunk_overlap)

    if splitter == "hybrid":
        return hybrid_chunk(text, chunk_size, chunk_overlap)

    if splitter == "sliding":
        return sliding_window_chunk(text, chunk_size, chunk_overlap)

    raise ValueError(f"Unsupported splitter mode: {splitter}")


# ---------------------------------------------------------------------------
# Chunk Quality Scoring
# ---------------------------------------------------------------------------

def score_chunk(chunk: str, ideal_length: int = 300) -> float:
    """
    Compute a quality score for a chunk.

    Factors:
    - Length score: penalize very short/very long
    - Density score: alphabetic density
    - Stopword score: penalize too many stopwords
    - Repetition penalty: penalize repeated characters/patterns

    Returns:
        float in [0, 1]
    """
    chunk = chunk.strip()
    if not chunk:
        return 0.0

    length = len(chunk)

    # Length score: ideal around ideal_length
    length_score = min(length / ideal_length, 1.0)

    # Alphabetic density
    alpha = sum(c.isalpha() for c in chunk)
    density_score = alpha / max(length, 1)

    # Stopword ratio (lower is better)
    words = chunk.split()
    stop_count = sum(1 for w in words if w.lower() in STOPWORDS)
    stopword_ratio = stop_count / max(len(words), 1)
    stopword_score = 1.0 - stopword_ratio  # invert: more stopwords → lower score

    # Repetition penalty
    repeats = len(re.findall(r"(.)\1{3,}", chunk))
    repetition_penalty = 1 - min(repeats / 5, 1)

    return (
        0.4 * length_score +
        0.3 * density_score +
        0.2 * stopword_score +
        0.1 * repetition_penalty
    )


def filter_low_quality(chunks: List[str], threshold: float = 0.4) -> List[str]:
    """
    Filter out chunks with quality score below threshold.
    """
    return [c for c in chunks if score_chunk(c) >= threshold]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def hash_chunk(chunk: str) -> str:
    """Return MD5 hash of a chunk."""
    return hashlib.md5(chunk.encode()).hexdigest()


def dedupe_exact(chunks: List[str]) -> List[str]:
    """
    Remove exact duplicate chunks using hashing.
    """
    seen = set()
    unique = []
    for c in chunks:
        h = hash_chunk(c)
        if h not in seen:
            seen.add(h)
            unique.append(c)
    return unique


def dedupe_semantic(chunks: List[str], embed_fn: Callable, threshold: float = 0.9) -> List[str]:
    """
    Remove near-duplicate chunks using embedding similarity.

    Optimized:
    - Single embedding call
    - Vectorized cosine similarity
    - O(n^2) but efficient for typical chunk counts (< 1000)
    """
    if not chunks:
        return []

    embeddings = np.array(embed_fn(chunks))
    keep_indices = []
    removed = set()

    for i in range(len(chunks)):
        if i in removed:
            continue

        keep_indices.append(i)

        sims = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings
        )[0]

        dupes = np.where(sims > threshold)[0]

        for d in dupes:
            if d != i:
                removed.add(d)

    return [chunks[i] for i in keep_indices]


async def dedupe_semantic_async(chunks, embed_fn):
    embeddings = await embed_fn(chunks)
    embeddings = np.array(embeddings)

    # your existing dedupe logic here
    unique = []
    seen = set()

    for i, emb in enumerate(embeddings):
        key = tuple(np.round(emb, 3))
        if key not in seen:
            seen.add(key)
            unique.append(chunks[i])

    return unique
