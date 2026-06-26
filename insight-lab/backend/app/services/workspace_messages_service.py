"""Workspace team chat — member-only, plain-text messages."""

from __future__ import annotations

from datetime import datetime, timezone

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import check_rate_limit
from app.core.exceptions import ForbiddenException, NotFoundException, RateLimitException
from app.core.yaml_config import get_yaml_config
from app.core.migration_guard import (
    PHASE3_017_MIGRATION_NOTICE,
    is_missing_team_chat_schema,
    run_or_raise_team_chat,
)
from app.services.team_chat_validation import validate_team_chat_body
from app.services.workspace_access import get_workspace_membership_role, require_workspace_role


async def _check_post_rate(user_id: str) -> None:
    cfg = get_yaml_config().team_chat
    allowed, retry_after = await check_rate_limit(
        key=f"team_chat:post:{user_id}",
        limit=cfg.post_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Message limit reached ({cfg.post_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )


async def _check_list_rate(user_id: str, workspace_id: str) -> None:
    cfg = get_yaml_config().team_chat
    allowed, retry_after = await check_rate_limit(
        key=f"team_chat:list:{user_id}:{workspace_id}",
        limit=cfg.list_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            "Too many chat refresh requests; try again shortly",
            retry_after=retry_after,
        )


async def _check_delete_rate(user_id: str) -> None:
    cfg = get_yaml_config().team_chat
    allowed, retry_after = await check_rate_limit(
        key=f"team_chat:delete:{user_id}",
        limit=cfg.delete_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Delete limit reached ({cfg.delete_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )


def _author_label(profile: dict | None) -> str:
    if not profile:
        return "Member"
    display = (profile.get("display_name") or "").strip()
    if display:
        return display
    email = (profile.get("email") or "").strip()
    if email and "@" in email:
        return email.split("@", 1)[0]
    return "Member"


def _serialize_message(row: dict, profile_map: dict[str, dict]) -> dict:
    author_id = row["author_id"]
    profile = profile_map.get(author_id)
    email = (profile.get("email") or "").strip() if profile else ""
    return {
        "id": row["id"],
        "workspace_id": row["workspace_id"],
        "author_id": author_id,
        "author_name": _author_label(profile),
        "author_email": email or None,
        "author_avatar_url": (profile.get("avatar_url") or "").strip() or None if profile else None,
        "body": row["body"],
        "created_at": row["created_at"],
        "is_own": False,
    }


async def list_workspace_messages(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    limit: int | None = None,
    before: str | None = None,
) -> dict:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    await _check_list_rate(user.id, workspace_id)

    cfg = get_yaml_config().team_chat
    page_size = min(limit or cfg.list_page_size, cfg.list_page_size)

    def _load() -> dict:
        query = (
            client.table("workspace_messages")
            .select("id, workspace_id, author_id, body, created_at")
            .eq("workspace_id", workspace_id)
            .is_("deleted_at", "null")
            .order("created_at", desc=True)
            .limit(page_size + 1)
        )
        if before:
            existing = (
                client.table("workspace_messages")
                .select("created_at")
                .eq("id", before)
                .eq("workspace_id", workspace_id)
                .limit(1)
                .execute()
            )
            if not existing.data:
                raise NotFoundException("Message not found")
            query = query.lt("created_at", existing.data[0]["created_at"])

        rows = query.execute().data or []
        has_more = len(rows) > page_size
        page_rows = rows[:page_size]

        author_ids = list({row["author_id"] for row in page_rows})
        profile_map: dict[str, dict] = {}
        if author_ids:
            profiles = (
                client.table("profiles")
                .select("id, display_name, email, avatar_url")
                .in_("id", author_ids)
                .execute()
                .data
                or []
            )
            profile_map = {row["id"]: row for row in profiles}

        messages = []
        for row in reversed(page_rows):
            payload = _serialize_message(row, profile_map)
            payload["is_own"] = row["author_id"] == user.id
            messages.append(payload)

        return {
            "workspace_id": workspace_id,
            "messages": messages,
            "has_more": has_more,
            "next_before": page_rows[-1]["id"] if has_more and page_rows else None,
        }

    try:
        return _load()
    except Exception as exc:
        if is_missing_team_chat_schema(exc):
            return {
                "workspace_id": workspace_id,
                "messages": [],
                "has_more": False,
                "next_before": None,
                "migration_required": True,
                "notice": PHASE3_017_MIGRATION_NOTICE,
            }
        raise


async def create_workspace_message(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    body: str,
) -> dict:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    await _check_post_rate(user.id)

    cfg = get_yaml_config().team_chat
    clean_body = validate_team_chat_body(body, max_length=cfg.message_max_length)

    def _insert() -> dict:
        inserted = (
            client.table("workspace_messages")
            .insert(
                {
                    "workspace_id": workspace_id,
                    "author_id": user.id,
                    "body": clean_body,
                }
            )
            .execute()
        )
        if not inserted.data:
            raise FileException("Could not send message", status_code=500)

        row = inserted.data[0]
        profile = (
            client.table("profiles")
            .select("id, display_name, email")
            .eq("id", user.id)
            .limit(1)
            .execute()
            .data
            or [None]
        )[0]
        payload = _serialize_message(row, {user.id: profile} if profile else {})
        payload["is_own"] = True
        return payload

    return run_or_raise_team_chat(_insert)


async def delete_workspace_message(
    client: Client,
    workspace_id: str,
    message_id: str,
    user: AuthUser,
) -> dict:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    await _check_delete_rate(user.id)

    result = (
        client.table("workspace_messages")
        .select("id, workspace_id, author_id, deleted_at")
        .eq("id", message_id)
        .eq("workspace_id", workspace_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Message not found")
    row = result.data[0]
    if row.get("deleted_at"):
        return {"deleted": True, "message_id": message_id}

    role = get_workspace_membership_role(client, workspace_id, user)
    is_owner = role == "owner"
    is_author = row["author_id"] == user.id
    if not is_author and not is_owner:
        raise ForbiddenException("You cannot delete this message")

    now = datetime.now(timezone.utc).isoformat()

    def _delete() -> dict:
        client.table("workspace_messages").update({"deleted_at": now, "deleted_by": user.id}).eq(
            "id", message_id
        ).eq("workspace_id", workspace_id).execute()
        return {"deleted": True, "message_id": message_id}

    return run_or_raise_team_chat(_delete)
