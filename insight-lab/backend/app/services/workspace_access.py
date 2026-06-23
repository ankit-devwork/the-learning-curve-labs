"""Workspace membership and document access for shared study sets."""

from __future__ import annotations

from typing import Literal

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.exceptions import ForbiddenException, NotFoundException

WorkspaceRole = Literal["owner", "editor", "viewer"]

ROLE_RANK: dict[str, int] = {"viewer": 0, "editor": 1, "owner": 2}


def _rank(role: str) -> int:
    return ROLE_RANK.get(role, -1)


def get_workspace_membership_role(
    client: Client,
    workspace_id: str,
    user: AuthUser,
) -> WorkspaceRole | None:
    result = (
        client.table("workspace_members")
        .select("role")
        .eq("workspace_id", workspace_id)
        .eq("user_id", user.id)
        .limit(1)
        .execute()
    )
    if result.data:
        return result.data[0]["role"]
    workspace = (
        client.table("workspaces")
        .select("owner_id")
        .eq("id", workspace_id)
        .limit(1)
        .execute()
    )
    if workspace.data and workspace.data[0]["owner_id"] == user.id:
        return "owner"
    return None


def require_workspace_role(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    min_role: WorkspaceRole = "viewer",
) -> WorkspaceRole:
    role = get_workspace_membership_role(client, workspace_id, user)
    if role is None or _rank(role) < _rank(min_role):
        raise ForbiddenException("You do not have access to this study set")
    return role


def get_workspace_row(client: Client, workspace_id: str, user: AuthUser) -> dict:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    result = (
        client.table("workspaces")
        .select("*")
        .eq("id", workspace_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Study set not found")
    row = result.data[0]
    role = get_workspace_membership_role(client, workspace_id, user)
    return {**row, "access_role": role}


def get_accessible_document(
    client: Client,
    document_id: str,
    user: AuthUser,
    *,
    min_role: WorkspaceRole = "viewer",
) -> dict:
    result = (
        client.table("documents")
        .select("*")
        .eq("id", document_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Document not found")
    doc = result.data[0]
    if doc["owner_id"] == user.id:
        return doc
    workspace_id = doc.get("workspace_id")
    if workspace_id:
        require_workspace_role(client, workspace_id, user, min_role=min_role)
        return doc
    raise NotFoundException("Document not found")


def can_edit_document(client: Client, document_id: str, user: AuthUser) -> bool:
    try:
        get_accessible_document(client, document_id, user, min_role="editor")
        return True
    except (NotFoundException, ForbiddenException):
        return False
