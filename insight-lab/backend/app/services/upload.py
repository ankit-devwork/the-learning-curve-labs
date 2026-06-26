import hashlib
import uuid

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import check_rate_limit
from app.core.exceptions import RateLimitException
from app.core.yaml_config import get_yaml_config
from app.services.cache_invalidation import invalidate_document_caches
from app.services.document_storage import (
    collect_document_storage_paths,
    remove_storage_paths,
    should_encrypt_uploads,
    upload_document_blob,
)
from app.services.graph_service import delete_document_from_neo4j
from app.services.upload_validation import ValidatedUpload
from app.services.workspace_access import require_editable_document

log = get_logger("upload")


def ensure_profile(client: Client, user: AuthUser) -> None:
    row: dict = {"id": user.id, "role": "user"}
    if user.email:
        row["email"] = user.email.strip().lower()
    client.table("profiles").upsert(row, on_conflict="id").execute()


def ensure_profile_and_workspace(client: Client, user: AuthUser) -> str:
    upload = get_yaml_config().upload
    ensure_profile(client, user)

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
    workspace_id = created.data[0]["id"]
    client.table("workspace_members").upsert(
        {"workspace_id": workspace_id, "user_id": user.id, "role": "owner"}
    ).execute()
    return workspace_id


def upload_document(
    client: Client,
    user: AuthUser,
    *,
    validated: ValidatedUpload,
    content: bytes,
    mime_type: str | None,
    workspace_id: str | None = None,
) -> dict:
    upload = get_yaml_config().upload
    resolved_workspace_id = workspace_id or ensure_profile_and_workspace(client, user)

    document_id = str(uuid.uuid4())
    storage_path = f"{user.id}/{document_id}/{validated.safe_filename}"
    file_hash = hashlib.sha256(content).hexdigest()
    storage_encrypted = False

    try:
        storage_encrypted = upload_document_blob(
            client,
            storage_path=storage_path,
            content=content,
            mime_type=mime_type,
        )
    except FileException:
        raise
    except Exception as exc:
        raise FileException(f"Storage upload failed: {exc}", status_code=502) from exc

    try:
        inserted = (
            client.table("documents")
            .insert(
                {
                    "id": document_id,
                    "workspace_id": resolved_workspace_id,
                    "owner_id": user.id,
                    "filename": validated.safe_filename,
                    "storage_path": storage_path,
                    "file_type": validated.file_type,
                    "mime_type": mime_type,
                    "file_hash": file_hash,
                    "status": "pending",
                    "storage_encrypted": storage_encrypted,
                }
            )
            .execute()
        )
    except Exception as exc:
        remove_storage_paths(client, [storage_path])
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
        storage_encrypted=storage_encrypted,
        status=row["status"],
    )
    return {
        "id": row["id"],
        "filename": row["filename"],
        "file_type": row["file_type"],
        "mime_type": row.get("mime_type"),
        "status": row["status"],
        "created_at": row["created_at"],
    }


async def delete_document(client: Client, document_id: str, user: AuthUser) -> dict:
    cfg = get_yaml_config().documents
    allowed, retry_after = await check_rate_limit(
        key=f"delete_document:{user.id}",
        limit=cfg.delete_rate_limit_per_hour,
        window_seconds=3600,
    )
    if not allowed:
        raise RateLimitException(
            f"Document delete limit reached ({cfg.delete_rate_limit_per_hour}/hour)",
            retry_after=retry_after,
        )

    doc = require_editable_document(client, document_id, user)
    workspace_id = doc["workspace_id"]
    file_hash = doc.get("file_hash")
    storage_paths = collect_document_storage_paths(client, document_id)

    client.table("documents").delete().eq("id", document_id).execute()
    remove_storage_paths(client, storage_paths)
    await delete_document_from_neo4j(document_id)
    await invalidate_document_caches(
        client,
        user_id=user.id,
        document_id=document_id,
        file_hash=file_hash,
    )

    log.info(
        "Document deleted",
        document_id=document_id,
        user_id=user.id,
        workspace_id=workspace_id,
        storage_paths_removed=len(storage_paths),
    )
    return {"deleted": True, "document_id": document_id}


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
    guidance = upload.guidance
    return {
        "max_bytes": upload.max_bytes,
        "max_mb": round(upload.max_bytes / (1024 * 1024), 2),
        "accept": upload.accept_attribute(),
        "allowed_extensions": sorted(upload.all_extensions()),
        "excel_extensions": upload.excel.extensions,
        "document_extensions": upload.document.extensions,
        "guidance": {
            "summary": guidance.summary,
            "points": guidance.points,
            "require_acknowledgment": guidance.require_acknowledgment,
        },
        "storage_encrypted": should_encrypt_uploads(),
    }
