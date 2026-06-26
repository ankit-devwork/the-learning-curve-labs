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
    PHASE3_022_MIGRATION_NOTICE,
    is_missing_team_chat_read_schema,
    is_missing_team_chat_schema,
    run_or_raise_team_chat,
)
from app.services.team_chat_validation import validate_team_chat_body
from app.services.workspace_access import get_workspace_membership_role, require_workspace_role
from app.services.workspace_service import list_workspaces


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


async def _check_inbox_rate(user_id: str) -> None:
    cfg = get_yaml_config().team_chat
    allowed, retry_after = await check_rate_limit(
        key=f"team_chat:inbox:{user_id}",
        limit=cfg.inbox_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            "Too many inbox refresh requests; try again shortly",
            retry_after=retry_after,
        )


async def _check_mark_read_rate(user_id: str, workspace_id: str) -> None:
    cfg = get_yaml_config().team_chat
    allowed, retry_after = await check_rate_limit(
        key=f"team_chat:mark_read:{user_id}:{workspace_id}",
        limit=cfg.mark_read_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            "Too many read-receipt updates; try again shortly",
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


def _attach_read_receipts(
    client: Client,
    user: AuthUser,
    messages: list[dict],
) -> None:
    own_ids = [message["id"] for message in messages if message.get("is_own")]
    if not own_ids:
        return

    reads = (
        client.table("workspace_message_reads")
        .select("message_id, user_id, read_at")
        .in_("message_id", own_ids)
        .neq("user_id", user.id)
        .execute()
        .data
        or []
    )
    if not reads:
        for message in messages:
            if message.get("is_own"):
                message["read_by_count"] = 0
                message["read_by"] = []
        return

    reader_ids = list({row["user_id"] for row in reads})
    profiles = (
        client.table("profiles")
        .select("id, display_name, email")
        .in_("id", reader_ids)
        .execute()
        .data
        or []
    )
    profile_map = {row["id"]: row for row in profiles}
    reads_by_message: dict[str, list[dict]] = {}
    for row in reads:
        reads_by_message.setdefault(row["message_id"], []).append(row)

    for message in messages:
        if not message.get("is_own"):
            continue
        message_reads = reads_by_message.get(message["id"], [])
        message["read_by_count"] = len(message_reads)
        message["read_by"] = [
            {
                "user_id": row["user_id"],
                "name": _author_label(profile_map.get(row["user_id"])),
                "read_at": row["read_at"],
            }
            for row in message_reads
        ]


def _count_unread_messages(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    last_read_at: str | None,
) -> int:
    query = (
        client.table("workspace_messages")
        .select("id", count="exact")
        .eq("workspace_id", workspace_id)
        .neq("author_id", user.id)
        .is_("deleted_at", "null")
    )
    if last_read_at:
        query = query.gt("created_at", last_read_at)
    result = query.execute()
    return int(result.count or 0)


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

        try:
            _attach_read_receipts(client, user, messages)
        except Exception as exc:
            if not is_missing_team_chat_read_schema(exc):
                raise

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


async def list_chat_inbox(client: Client, user: AuthUser) -> dict:
    """All study-sheet conversations for the current user with preview and unread counts."""
    await _check_inbox_rate(user.id)

    def _load() -> dict:
        workspaces = list_workspaces(client, user)
        if not workspaces:
            return {"conversations": [], "total_unread": 0}

        workspace_ids = [row["id"] for row in workspaces]
        read_rows = (
            client.table("workspace_chat_read_state")
            .select("workspace_id, last_read_at, last_read_message_id")
            .eq("user_id", user.id)
            .in_("workspace_id", workspace_ids)
            .execute()
            .data
            or []
        )
        read_map = {row["workspace_id"]: row for row in read_rows}

        recent_messages = (
            client.table("workspace_messages")
            .select("id, workspace_id, author_id, body, created_at")
            .in_("workspace_id", workspace_ids)
            .is_("deleted_at", "null")
            .order("created_at", desc=True)
            .limit(min(len(workspace_ids) * 5, 200))
            .execute()
            .data
            or []
        )

        latest_by_workspace: dict[str, dict] = {}
        author_ids: set[str] = set()
        for row in recent_messages:
            workspace_id = row["workspace_id"]
            if workspace_id not in latest_by_workspace:
                latest_by_workspace[workspace_id] = row
                author_ids.add(row["author_id"])

        profile_map: dict[str, dict] = {}
        if author_ids:
            profiles = (
                client.table("profiles")
                .select("id, display_name, email, avatar_url")
                .in_("id", list(author_ids))
                .execute()
                .data
                or []
            )
            profile_map = {row["id"]: row for row in profiles}

        conversations = []
        total_unread = 0
        for workspace in workspaces:
            workspace_id = workspace["id"]
            read_state = read_map.get(workspace_id)
            last_read_at = read_state.get("last_read_at") if read_state else None
            unread_count = _count_unread_messages(client, workspace_id, user, last_read_at)
            total_unread += unread_count

            latest = latest_by_workspace.get(workspace_id)
            last_message = None
            if latest:
                last_message = {
                    "id": latest["id"],
                    "body": latest["body"],
                    "created_at": latest["created_at"],
                    "author_id": latest["author_id"],
                    "author_name": _author_label(profile_map.get(latest["author_id"])),
                    "is_own": latest["author_id"] == user.id,
                }

            conversations.append(
                {
                    "workspace_id": workspace_id,
                    "workspace_name": workspace["name"],
                    "access_role": workspace.get("access_role"),
                    "is_owner": workspace.get("is_owner", False),
                    "unread_count": unread_count,
                    "last_message": last_message,
                    "last_message_at": last_message["created_at"] if last_message else None,
                }
            )

        conversations.sort(
            key=lambda row: row["last_message_at"] or "",
            reverse=True,
        )
        return {"conversations": conversations, "total_unread": total_unread}

    try:
        return _load()
    except Exception as exc:
        if is_missing_team_chat_schema(exc) or is_missing_team_chat_read_schema(exc):
            return {
                "conversations": [],
                "total_unread": 0,
                "migration_required": True,
                "notice": PHASE3_022_MIGRATION_NOTICE,
            }
        raise


async def mark_workspace_messages_read(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    up_to_message_id: str | None = None,
) -> dict:
    require_workspace_role(client, workspace_id, user, min_role="viewer")
    await _check_mark_read_rate(user.id, workspace_id)

    def _mark() -> dict:
        if up_to_message_id:
            anchor = (
                client.table("workspace_messages")
                .select("id, created_at")
                .eq("id", up_to_message_id)
                .eq("workspace_id", workspace_id)
                .is_("deleted_at", "null")
                .limit(1)
                .execute()
            )
            if not anchor.data:
                raise NotFoundException("Message not found")
            anchor_row = anchor.data[0]
        else:
            anchor = (
                client.table("workspace_messages")
                .select("id, created_at")
                .eq("workspace_id", workspace_id)
                .is_("deleted_at", "null")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if not anchor.data:
                now = datetime.now(timezone.utc).isoformat()
                client.table("workspace_chat_read_state").upsert(
                    {
                        "user_id": user.id,
                        "workspace_id": workspace_id,
                        "last_read_at": now,
                        "last_read_message_id": None,
                        "updated_at": now,
                    },
                    on_conflict="user_id,workspace_id",
                ).execute()
                return {"marked_read": True, "last_read_message_id": None, "unread_count": 0}
            anchor_row = anchor.data[0]

        now = datetime.now(timezone.utc).isoformat()
        client.table("workspace_chat_read_state").upsert(
            {
                "user_id": user.id,
                "workspace_id": workspace_id,
                "last_read_at": anchor_row["created_at"],
                "last_read_message_id": anchor_row["id"],
                "updated_at": now,
            },
            on_conflict="user_id,workspace_id",
        ).execute()

        unread_candidates = (
            client.table("workspace_messages")
            .select("id")
            .eq("workspace_id", workspace_id)
            .neq("author_id", user.id)
            .is_("deleted_at", "null")
            .lte("created_at", anchor_row["created_at"])
            .execute()
            .data
            or []
        )
        for row in unread_candidates:
            client.table("workspace_message_reads").upsert(
                {
                    "message_id": row["id"],
                    "user_id": user.id,
                    "read_at": now,
                },
                on_conflict="message_id,user_id",
            ).execute()

        unread_count = _count_unread_messages(client, workspace_id, user, anchor_row["created_at"])
        return {
            "marked_read": True,
            "last_read_message_id": anchor_row["id"],
            "unread_count": unread_count,
        }

    try:
        return _mark()
    except NotFoundException:
        raise
    except Exception as exc:
        if is_missing_team_chat_read_schema(exc):
            from pycorekit.exceptions.file import FileException

            raise FileException(PHASE3_022_MIGRATION_NOTICE, status_code=503) from exc
        raise
