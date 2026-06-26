"""Upload, download, and purge document blobs in Supabase Storage."""

from __future__ import annotations

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.storage_crypto import decrypt_bytes, encrypt_bytes, storage_encryption_enabled
from app.core.yaml_config import get_yaml_config

log = get_logger("document_storage")


def _bucket(client: Client) -> str:
    return get_yaml_config().upload.storage_bucket


def should_encrypt_uploads() -> bool:
    return storage_encryption_enabled()


def upload_document_blob(
    client: Client,
    *,
    storage_path: str,
    content: bytes,
    mime_type: str | None,
    encrypt: bool | None = None,
) -> bool:
    """Store a document blob. Returns True when content was encrypted."""
    use_encryption = should_encrypt_uploads() if encrypt is None else encrypt
    payload = encrypt_bytes(content) if use_encryption else content

    try:
        client.storage.from_(_bucket(client)).upload(
            storage_path,
            payload,
            file_options={
                "content-type": mime_type or "application/octet-stream",
                "upsert": "false",
            },
        )
    except Exception as exc:
        raise FileException(f"Storage upload failed: {exc}", status_code=502) from exc

    return use_encryption


def download_document_blob(client: Client, document: dict) -> bytes:
    """Download and optionally decrypt a document blob."""
    try:
        raw = client.storage.from_(_bucket(client)).download(document["storage_path"])
    except Exception as exc:
        raise FileException(f"Failed to download document from storage: {exc}", status_code=502) from exc

    if document.get("storage_encrypted"):
        return decrypt_bytes(raw)
    return raw


def remove_storage_paths(client: Client, paths: list[str]) -> None:
    """Best-effort removal of storage objects."""
    cleaned = [path for path in paths if path]
    if not cleaned:
        return
    try:
        client.storage.from_(_bucket(client)).remove(cleaned)
    except Exception as exc:
        log.warning("Storage cleanup failed", paths=cleaned, error=str(exc))


def collect_document_storage_paths(client: Client, document_id: str) -> list[str]:
    """Gather primary and derived storage paths for a document."""
    paths: list[str] = []

    doc_rows = (
        client.table("documents")
        .select("storage_path")
        .eq("id", document_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if doc_rows and doc_rows[0].get("storage_path"):
        paths.append(doc_rows[0]["storage_path"])

    try:
        audio_rows = (
            client.table("document_audio_overviews")
            .select("storage_path")
            .eq("document_id", document_id)
            .execute()
            .data
            or []
        )
        for row in audio_rows:
            if row.get("storage_path"):
                paths.append(row["storage_path"])
    except Exception:
        pass

    return paths


def collect_workspace_storage_paths(client: Client, workspace_id: str) -> list[str]:
    """Gather all document and derived storage paths in a workspace."""
    docs = (
        client.table("documents")
        .select("id")
        .eq("workspace_id", workspace_id)
        .execute()
        .data
        or []
    )
    paths: list[str] = []
    for doc in docs:
        paths.extend(collect_document_storage_paths(client, doc["id"]))
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique
