import re

SOURCE_PREVIEW_MAX = 420


def make_source_preview(content: str, *, max_len: int = SOURCE_PREVIEW_MAX) -> str:
    collapsed = " ".join(content.split())
    if len(collapsed) <= max_len:
        return collapsed

    excerpt = collapsed[:max_len]
    for separator in (". ", "? ", "! ", "; "):
        boundary = excerpt.rfind(separator)
        if boundary >= int(max_len * 0.55):
            return excerpt[: boundary + len(separator)].strip()

    word_boundary = excerpt.rfind(" ")
    if word_boundary >= int(max_len * 0.65):
        return excerpt[:word_boundary].rstrip() + "…"

    return excerpt.rstrip() + "…"


def collapse_sources_by_document(sources: list[dict]) -> list[dict]:
    """Keep one display row per document, preferring the strongest match."""
    best_by_document: dict[str, dict] = {}
    for source in sources:
        document_id = source["document_id"]
        existing = best_by_document.get(document_id)
        if existing is None:
            best_by_document[document_id] = source
            continue
        current_score = source.get("similarity") or 0
        existing_score = existing.get("similarity") or 0
        if current_score > existing_score:
            best_by_document[document_id] = source

    ordered: list[dict] = []
    seen: set[str] = set()
    for source in sources:
        document_id = source["document_id"]
        if document_id in seen:
            continue
        seen.add(document_id)
        ordered.append(best_by_document[document_id])
    return ordered


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


def hash_document_ids(document_ids: list[str]) -> str:
    import hashlib

    joined = ",".join(sorted(document_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
