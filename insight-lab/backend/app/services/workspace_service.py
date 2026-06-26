import re
from uuid import uuid4

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.exceptions import ForbiddenException, NotFoundException
from app.services.document_storage import collect_workspace_storage_paths, remove_storage_paths
from app.services.upload import ensure_profile_and_workspace
from app.services.workspace_access import (
    get_workspace_membership_role,
    get_workspace_row,
    require_workspace_role,
)

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
    require_workspace_role(client, workspace_id, user, min_role="owner")
    result = (
        client.table("workspaces")
        .select("*")
        .eq("id", workspace_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Study set not found")
    return result.data[0]


def list_workspaces(client: Client, user: AuthUser) -> list[dict]:
    ensure_profile_and_workspace(client, user)
    memberships = (
        client.table("workspace_members")
        .select("workspace_id, role")
        .eq("user_id", user.id)
        .execute()
        .data
        or []
    )
    workspace_ids = [row["workspace_id"] for row in memberships]
    if not workspace_ids:
        return []

    role_map = {row["workspace_id"]: row["role"] for row in memberships}
    result = (
        client.table("workspaces")
        .select("id, name, description, created_at, updated_at, owner_id")
        .in_("id", workspace_ids)
        .order("created_at")
        .execute()
    )
    rows = result.data or []
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "description": row.get("description"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "access_role": role_map.get(row["id"]),
            "is_owner": row["owner_id"] == user.id,
            "shared": row["owner_id"] != user.id,
        }
        for row in rows
    ]


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
    workspace_id = str(uuid4())
    row = {
        "id": workspace_id,
        "owner_id": user.id,
        "name": safe_name,
        "description": safe_description,
    }
    inserted = client.table("workspaces").insert(row).execute()
    if not inserted.data:
        raise FileException("Failed to create study set", status_code=500)
    client.table("workspace_members").insert(
        {
            "workspace_id": workspace_id,
            "user_id": user.id,
            "role": "owner",
        }
    ).execute()
    created = inserted.data[0]
    return {
        "id": created["id"],
        "name": created["name"],
        "description": created.get("description"),
        "created_at": created["created_at"],
        "updated_at": created["updated_at"],
        "access_role": "owner",
        "is_owner": True,
        "shared": False,
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
        .execute()
    )
    if not updated.data:
        raise FileException("Failed to update study set", status_code=500)
    return get_workspace(client, workspace_id, user)


def delete_workspace(client: Client, workspace_id: str, user: AuthUser) -> None:
    workspace = _get_owned_workspace(client, workspace_id, user)
    owned_sets = (
        client.table("workspaces")
        .select("id")
        .eq("owner_id", user.id)
        .execute()
        .data
        or []
    )
    if len(owned_sets) <= 1:
        raise FileException("You must keep at least one study set", status_code=409)
    storage_paths = collect_workspace_storage_paths(client, workspace["id"])
    client.table("workspaces").delete().eq("id", workspace["id"]).execute()
    remove_storage_paths(client, storage_paths)


def get_workspace(client: Client, workspace_id: str, user: AuthUser) -> dict:
    row = get_workspace_row(client, workspace_id, user)
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row.get("description"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "access_role": row.get("access_role"),
        "is_owner": row["owner_id"] == user.id,
        "shared": row["owner_id"] != user.id,
    }


def list_workspace_documents(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    limit: int = 50,
) -> list[dict]:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    limit = min(max(limit, 1), 100)
    result = (
        client.table("documents")
        .select("id, filename, file_type, mime_type, status, created_at, workspace_id, owner_id")
        .eq("workspace_id", workspace_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_workspace_stats(client: Client, workspace_id: str, user: AuthUser) -> dict:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    docs = (
        client.table("documents")
        .select("id, status, file_type")
        .eq("workspace_id", workspace_id)
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
        require_workspace_role(client, workspace_id, user, min_role="editor")
        return workspace_id
    return ensure_profile_and_workspace(client, user)
