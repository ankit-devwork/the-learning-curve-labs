import re

SOURCE_PREVIEW_MAX = 280


def make_source_preview(content: str, *, max_len: int = SOURCE_PREVIEW_MAX) -> str:
    collapsed = " ".join(content.split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1].rstrip() + "…"


def strip_excerpt_markers(answer: str) -> str:
    cleaned = re.sub(r"\s*\[excerpt_\d+\]\s*", " ", answer, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[chunk\s+\d+\]\s*", " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split()).strip()


def build_source_citations(
    rows: list[dict],
    *,
    filename: str | None = None,
    document_id: str | None = None,
) -> list[dict]:
    sources: list[dict] = []
    for index, row in enumerate(rows, start=1):
        doc_id = row.get("document_id") or document_id
        name = row.get("filename") or filename or "Document"
        sources.append(
            {
                "id": index,
                "document_id": doc_id,
                "filename": name,
                "chunk_index": row.get("chunk_index"),
                "preview": make_source_preview(row["content"]),
                "similarity": row.get("similarity"),
            }
        )
    return sources


def source_ref_key(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index}"


def hash_source_refs(refs: list[dict]) -> str:
    import hashlib

    parts = sorted(f"{row['document_id']}:{row['chunk_index']}" for row in refs)
    joined = ",".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
