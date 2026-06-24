"""Per-member learning analytics for shared study sheets (owner/editor view)."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from supabase import Client

from app.core.auth import AuthUser
from app.core.yaml_config import get_yaml_config
from app.services.sharing_service import list_workspace_members
from app.services.workspace_access import require_workspace_role


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _avg_percent(scores: list[tuple[int, int]]) -> float | None:
    percents = [(s / t) * 100 for s, t in scores if t > 0]
    if not percents:
        return None
    return round(sum(percents) / len(percents), 1)


def get_classroom_analytics(client: Client, workspace_id: str, user: AuthUser) -> dict[str, Any]:
    """Aggregate quiz, mastery, and flashcard activity per workspace member."""
    require_workspace_role(client, workspace_id, user, min_role="editor")

    docs = (
        client.table("documents")
        .select("id, filename, file_type, status")
        .eq("workspace_id", workspace_id)
        .execute()
        .data
        or []
    )
    document_ids = [row["id"] for row in docs if row.get("file_type") == "document"]
    quiz_ids: list[str] = []
    if document_ids:
        quiz_rows = (
            client.table("quizzes")
            .select("id, document_id, title, published, public_share_token")
            .in_("document_id", document_ids)
            .execute()
            .data
            or []
        )
        quiz_ids = [row["id"] for row in quiz_rows]

    members = list_workspace_members(client, workspace_id, user)
    member_ids = [row["user_id"] for row in members]

    quiz_by_user: dict[str, list[tuple[int, int]]] = defaultdict(list)
    quiz_last: dict[str, datetime | None] = {uid: None for uid in member_ids}
    if quiz_ids and member_ids:
        attempts = (
            client.table("quiz_attempts")
            .select("user_id, score, total, completed_at")
            .in_("quiz_id", quiz_ids)
            .in_("user_id", member_ids)
            .execute()
            .data
            or []
        )
        for row in attempts:
            uid = row["user_id"]
            quiz_by_user[uid].append((int(row["score"]), int(row["total"])))
            ts = _parse_ts(row.get("completed_at"))
            if ts and (quiz_last[uid] is None or ts > quiz_last[uid]):
                quiz_last[uid] = ts

    mastery_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mastery_last: dict[str, datetime | None] = {uid: None for uid in member_ids}
    if document_ids and member_ids:
        try:
            mastery_rows = (
                client.table("concept_mastery")
                .select("user_id, attempts, correct, last_attempt_at, concept_id")
                .in_("document_id", document_ids)
                .in_("user_id", member_ids)
                .execute()
                .data
                or []
            )
            for row in mastery_rows:
                uid = row["user_id"]
                attempts = int(row.get("attempts") or 0)
                correct = int(row.get("correct") or 0)
                percent = round((correct / attempts) * 100, 1) if attempts > 0 else None
                mastery_by_user[uid].append({**row, "percent": percent})
                ts = _parse_ts(row.get("last_attempt_at"))
                if ts and (mastery_last[uid] is None or ts > mastery_last[uid]):
                    mastery_last[uid] = ts
        except Exception:
            pass

    flashcard_by_user: dict[str, list[bool]] = defaultdict(list)
    flashcard_last: dict[str, datetime | None] = {uid: None for uid in member_ids}
    if document_ids and member_ids:
        sets = (
            client.table("flashcard_sets")
            .select("id")
            .in_("document_id", document_ids)
            .execute()
            .data
            or []
        )
        set_ids = [row["id"] for row in sets]
        if set_ids:
            cards = (
                client.table("flashcards")
                .select("id")
                .in_("set_id", set_ids)
                .execute()
                .data
                or []
            )
            card_ids = [row["id"] for row in cards]
            if card_ids:
                reviews = (
                    client.table("flashcard_reviews")
                    .select("user_id, knew, reviewed_at")
                    .in_("flashcard_id", card_ids)
                    .in_("user_id", member_ids)
                    .execute()
                    .data
                    or []
                )
                for row in reviews:
                    uid = row["user_id"]
                    flashcard_by_user[uid].append(bool(row.get("knew")))
                    ts = _parse_ts(row.get("reviewed_at"))
                    if ts and (flashcard_last[uid] is None or ts > flashcard_last[uid]):
                        flashcard_last[uid] = ts

    cfg = get_yaml_config().adaptive_quiz
    weak_threshold = cfg.weak_threshold_percent

    member_stats: list[dict[str, Any]] = []
    class_quiz_scores: list[tuple[int, int]] = []

    for member in members:
        uid = member["user_id"]
        scores = quiz_by_user.get(uid, [])
        class_quiz_scores.extend(scores)
        mastery_rows = mastery_by_user.get(uid, [])
        mastery_percents = [r["percent"] for r in mastery_rows if r.get("percent") is not None]
        weak_count = sum(1 for p in mastery_percents if p is not None and p < weak_threshold)
        flashcard_reviews = flashcard_by_user.get(uid, [])
        knew_count = sum(1 for knew in flashcard_reviews if knew)

        last_activity: datetime | None = None
        for candidate in (quiz_last.get(uid), mastery_last.get(uid), flashcard_last.get(uid)):
            if candidate and (last_activity is None or candidate > last_activity):
                last_activity = candidate

        member_stats.append(
            {
                "user_id": uid,
                "email": member.get("email"),
                "full_name": member.get("full_name"),
                "role": member.get("role"),
                "joined_at": member.get("joined_at"),
                "quiz_attempts": len(scores),
                "avg_quiz_percent": _avg_percent(scores),
                "flashcard_reviews": len(flashcard_reviews),
                "flashcard_knew_percent": round((knew_count / len(flashcard_reviews)) * 100, 1)
                if flashcard_reviews
                else None,
                "mastery_avg_percent": round(sum(mastery_percents) / len(mastery_percents), 1)
                if mastery_percents
                else None,
                "weak_topic_count": weak_count,
                "last_activity_at": last_activity.isoformat() if last_activity else None,
            }
        )

    member_stats.sort(
        key=lambda row: (
            row.get("last_activity_at") is None,
            row.get("last_activity_at") or "",
        ),
        reverse=True,
    )

    public_attempts = 0
    public_avg_percent: float | None = None
    if quiz_ids:
        try:
            public_rows = (
                client.table("quiz_public_attempts")
                .select("score, total")
                .in_("quiz_id", quiz_ids)
                .execute()
                .data
                or []
            )
            public_attempts = len(public_rows)
            public_avg_percent = _avg_percent(
                [(int(r["score"]), int(r["total"])) for r in public_rows if r.get("total")]
            )
        except Exception:
            pass

    return {
        "workspace_id": workspace_id,
        "member_count": len(members),
        "ready_documents": sum(1 for d in docs if d.get("status") == "ready"),
        "class_avg_quiz_percent": _avg_percent(class_quiz_scores),
        "public_quiz_attempts": public_attempts,
        "public_quiz_avg_percent": public_avg_percent,
        "members": member_stats,
    }
