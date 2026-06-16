import hashlib
import uuid

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.yaml_config import get_yaml_config
from app.services.upload_validation import ValidatedUpload

log = get_logger("upload")


def ensure_profile_and_workspace(client: Client, user: AuthUser) -> str:
    upload = get_yaml_config().upload
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
        .insert({"owner_id": user.id, "name": upload.default_workspace_name})
        .execute()
    )
    if not created.data:
        raise FileException("Failed to create workspace", status_code=500)
    return created.data[0]["id"]


def upload_document(
    client: Client,
    user: AuthUser,
    *,
    validated: ValidatedUpload,
    content: bytes,
    mime_type: str | None,
) -> dict:
    upload = get_yaml_config().upload
    workspace_id = ensure_profile_and_workspace(client, user)

    document_id = str(uuid.uuid4())
    storage_path = f"{user.id}/{document_id}/{validated.safe_filename}"
    file_hash = hashlib.sha256(content).hexdigest()

    try:
        client.storage.from_(upload.storage_bucket).upload(
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
                    "filename": validated.safe_filename,
                    "storage_path": storage_path,
                    "file_type": validated.file_type,
                    "mime_type": mime_type,
                    "file_hash": file_hash,
                    "status": "pending",
                }
            )
            .execute()
        )
    except Exception as exc:
        client.storage.from_(upload.storage_bucket).remove([storage_path])
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


def list_documents(client: Client, user: AuthUser, *, limit: int | None = None) -> list[dict]:
    upload = get_yaml_config().upload
    page_size = limit if limit is not None else upload.documents_list_default_limit
    result = (
        client.table("documents")
        .select("id, filename, file_type, mime_type, status, created_at")
        .eq("owner_id", user.id)
        .order("created_at", desc=True)
        .limit(page_size)
        .execute()
    )
    return result.data or []


def get_upload_public_config() -> dict:
    upload = get_yaml_config().upload
    return {
        "max_bytes": upload.max_bytes,
        "max_mb": round(upload.max_bytes / (1024 * 1024), 2),
        "accept": upload.accept_attribute(),
        "allowed_extensions": sorted(upload.all_extensions()),
        "excel_extensions": upload.excel.extensions,
        "document_extensions": upload.document.extensions,
    }
