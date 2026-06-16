import hashlib
import re
import uuid
from pathlib import Path

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.config import settings

log = get_logger("upload")

EXCEL_EXTENSIONS = {".xlsx", ".xls", ".csv"}
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc"}

ALLOWED_EXTENSIONS = EXCEL_EXTENSIONS | DOCUMENT_EXTENSIONS


def _safe_filename(name: str) -> str:
    base = Path(name).name.strip()
    if not base:
        raise FileException("Filename is required")
    cleaned = re.sub(r"[^\w.\- ]+", "_", base)
    return cleaned[:255]


def detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in EXCEL_EXTENSIONS:
        return "excel"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
    raise FileException(f"Unsupported file type. Allowed: {allowed}")


def ensure_profile_and_workspace(client: Client, user: AuthUser) -> str:
    client.table("profiles").upsert(
        {"id": user.id, "role": "user"},
        on_conflict="id",
    ).execute()

    existing = (
        client.table("workspaces")
        .select("id")
        .eq("owner_id", user.id)
        .order("created_at")
        .limit(1)
        .execute()
    )
    if existing.data:
        return existing.data[0]["id"]

    created = (
        client.table("workspaces")
        .insert({"owner_id": user.id, "name": "My Workspace"})
        .execute()
    )
    if not created.data:
        raise FileException("Failed to create workspace", status_code=500)
    return created.data[0]["id"]


def upload_document(
    client: Client,
    user: AuthUser,
    *,
    filename: str,
    content: bytes,
    mime_type: str | None,
) -> dict:
    if len(content) > settings.upload_max_bytes:
        max_mb = settings.upload_max_bytes // (1024 * 1024)
        raise FileException(f"File too large. Maximum size is {max_mb} MB")

    if not content:
        raise FileException("Uploaded file is empty")

    safe_name = _safe_filename(filename)
    file_type = detect_file_type(safe_name)
    workspace_id = ensure_profile_and_workspace(client, user)

    document_id = str(uuid.uuid4())
    storage_path = f"{user.id}/{document_id}/{safe_name}"
    file_hash = hashlib.sha256(content).hexdigest()

    try:
        client.storage.from_(settings.storage_bucket).upload(
            storage_path,
            content,
            file_options={
                "content-type": mime_type or "application/octet-stream",
                "upsert": "false",
            },
        )
    except Exception as exc:
        raise FileException(f"Storage upload failed: {exc}", status_code=502) from exc

    try:
        inserted = (
            client.table("documents")
            .insert(
                {
                    "id": document_id,
                    "workspace_id": workspace_id,
                    "owner_id": user.id,
                    "filename": safe_name,
                    "storage_path": storage_path,
                    "file_type": file_type,
                    "mime_type": mime_type,
                    "file_hash": file_hash,
                    "status": "pending",
                }
            )
            .execute()
        )
    except Exception as exc:
        client.storage.from_(settings.storage_bucket).remove([storage_path])
        raise FileException(f"Failed to save document metadata: {exc}", status_code=500) from exc

    if not inserted.data:
        raise FileException("Failed to save document metadata", status_code=500)

    row = inserted.data[0]
    log.info(
        "Document uploaded",
        document_id=row["id"],
        user_id=user.id,
        filename=row["filename"],
        file_type=row["file_type"],
        storage_path=row["storage_path"],
        status=row["status"],
    )
    return {
        "id": row["id"],
        "filename": row["filename"],
        "file_type": row["file_type"],
        "mime_type": row.get("mime_type"),
        "status": row["status"],
        "storage_path": row["storage_path"],
        "created_at": row["created_at"],
    }


def list_documents(client: Client, user: AuthUser, *, limit: int = 20) -> list[dict]:
    result = (
        client.table("documents")
        .select("id, filename, file_type, mime_type, status, created_at")
        .eq("owner_id", user.id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []
