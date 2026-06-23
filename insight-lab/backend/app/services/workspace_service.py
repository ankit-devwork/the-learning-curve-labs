import re
from uuid import uuid4

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.exceptions import NotFoundException
from app.services.upload import ensure_profile_and_workspace

WORKSPACE_NAME_MAX = 100
WORKSPACE_NAME_MIN = 1


def _sanitize_workspace_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name.strip())
    if len(cleaned) < WORKSPACE_NAME_MIN:
        raise FileException("Study set name is required")
    if len(cleaned) > WORKSPACE_NAME_MAX:
        raise FileException(f"Study set name must be at most {WORKSPACE_NAME_MAX} characters")
    return cleaned


def _get_owned_workspace(client: Client, workspace_id: str, user: AuthUser) -> dict:
    result = (
        client.table("workspaces")
        .select("*")
        .eq("id", workspace_id)
        .eq("owner_id", user.id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Study set not found")
    return result.data[0]


def list_workspaces(client: Client, user: AuthUser) -> list[dict]:
    ensure_profile_and_workspace(client, user)
    result = (
        client.table("workspaces")
        .select("id, name, description, created_at, updated_at")
        .eq("owner_id", user.id)
        .order("created_at")
        .execute()
    )
    return result.data or []


def create_workspace(
    client: Client,
    user: AuthUser,
    *,
    name: str,
    description: str | None = None,
) -> dict:
    ensure_profile_and_workspace(client, user)
    safe_name = _sanitize_workspace_name(name)
    safe_description = description.strip()[:500] if description else None
    row = {
        "id": str(uuid4()),
        "owner_id": user.id,
        "name": safe_name,
        "description": safe_description,
    }
    inserted = client.table("workspaces").insert(row).execute()
    if not inserted.data:
        raise FileException("Failed to create study set", status_code=500)
    created = inserted.data[0]
    return {
        "id": created["id"],
        "name": created["name"],
        "description": created.get("description"),
        "created_at": created["created_at"],
        "updated_at": created["updated_at"],
    }


def update_workspace(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    name: str | None = None,
    description: str | None = None,
) -> dict:
    _get_owned_workspace(client, workspace_id, user)
    updates: dict = {}
    if name is not None:
        updates["name"] = _sanitize_workspace_name(name)
    if description is not None:
        updates["description"] = description.strip()[:500] if description.strip() else None
    if not updates:
        return get_workspace(client, workspace_id, user)
    updated = (
        client.table("workspaces")
        .update(updates)
        .eq("id", workspace_id)
        .eq("owner_id", user.id)
        .execute()
    )
    if not updated.data:
        raise FileException("Failed to update study set", status_code=500)
    row = updated.data[0]
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row.get("description"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def delete_workspace(client: Client, workspace_id: str, user: AuthUser) -> None:
    workspace = _get_owned_workspace(client, workspace_id, user)
    all_sets = (
        client.table("workspaces")
        .select("id")
        .eq("owner_id", user.id)
        .execute()
        .data
        or []
    )
    if len(all_sets) <= 1:
        raise FileException("You must keep at least one study set", status_code=409)
    client.table("workspaces").delete().eq("id", workspace["id"]).eq("owner_id", user.id).execute()


def get_workspace(client: Client, workspace_id: str, user: AuthUser) -> dict:
    row = _get_owned_workspace(client, workspace_id, user)
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row.get("description"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_workspace_documents(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    limit: int = 50,
) -> list[dict]:
    _get_owned_workspace(client, workspace_id, user)
    limit = min(max(limit, 1), 100)
    result = (
        client.table("documents")
        .select("id, filename, file_type, mime_type, status, created_at, workspace_id")
        .eq("workspace_id", workspace_id)
        .eq("owner_id", user.id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_workspace_stats(client: Client, workspace_id: str, user: AuthUser) -> dict:
    _get_owned_workspace(client, workspace_id, user)
    docs = (
        client.table("documents")
        .select("id, status, file_type")
        .eq("workspace_id", workspace_id)
        .eq("owner_id", user.id)
        .execute()
        .data
        or []
    )
    doc_ids = [row["id"] for row in docs if row["file_type"] == "document"]
    ready_count = sum(1 for row in docs if row["status"] == "ready")
    quiz_attempts = 0
    avg_quiz_percent: float | None = None
    if doc_ids:
        quizzes = (
            client.table("quizzes")
            .select("id")
            .in_("document_id", doc_ids)
            .execute()
            .data
            or []
        )
        quiz_ids = [row["id"] for row in quizzes]
        if quiz_ids:
            attempts = (
                client.table("quiz_attempts")
                .select("score, total")
                .eq("user_id", user.id)
                .in_("quiz_id", quiz_ids)
                .execute()
                .data
                or []
            )
            quiz_attempts = len(attempts)
            if attempts:
                percents = [
                    (row["score"] / row["total"]) * 100
                    for row in attempts
                    if row.get("total", 0) > 0
                ]
                if percents:
                    avg_quiz_percent = round(sum(percents) / len(percents), 1)
    return {
        "workspace_id": workspace_id,
        "document_count": len(docs),
        "ready_count": ready_count,
        "document_files": sum(1 for row in docs if row["file_type"] == "document"),
        "excel_files": sum(1 for row in docs if row["file_type"] == "excel"),
        "quiz_attempts": quiz_attempts,
        "avg_quiz_percent": avg_quiz_percent,
    }


def resolve_workspace_id(
    client: Client,
    user: AuthUser,
    workspace_id: str | None,
) -> str:
    if workspace_id:
        return _get_owned_workspace(client, workspace_id, user)["id"]
    return ensure_profile_and_workspace(client, user)
