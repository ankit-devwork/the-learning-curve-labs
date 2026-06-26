"""Simple spaced repetition scheduling for flashcards."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from supabase import Client

from app.core.auth import AuthUser
from app.core.migration_guard import run_or_none_phase14


def _now() -> datetime:
    return datetime.now(timezone.utc)


def update_srs_after_review(
    client: Client,
    user: AuthUser,
    *,
    flashcard_id: str,
    knew: bool,
) -> dict[str, Any]:
    existing = (
        client.table("flashcard_srs_states")
        .select("*")
        .eq("user_id", user.id)
        .eq("flashcard_id", flashcard_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    now = _now()
    if existing:
        state = existing[0]
        interval = int(state.get("interval_days") or 1)
        ease = float(state.get("ease_factor") or 2.5)
        reps = int(state.get("repetitions") or 0)
    else:
        interval = 1
        ease = 2.5
        reps = 0

    if knew:
        reps += 1
        ease = min(3.0, round(ease + 0.1, 2))
        if reps == 1:
            interval = 1
        elif reps == 2:
            interval = 3
        else:
            interval = max(1, round(interval * ease))
    else:
        reps = 0
        ease = max(1.3, round(ease - 0.2, 2))
        interval = 1

    due_at = now + timedelta(days=interval)
    payload = {
        "user_id": user.id,
        "flashcard_id": flashcard_id,
        "interval_days": interval,
        "due_at": due_at.isoformat(),
        "ease_factor": ease,
        "repetitions": reps,
        "updated_at": now.isoformat(),
    }
    client.table("flashcard_srs_states").upsert(payload).execute()
    return {
        "flashcard_id": flashcard_id,
        "due_at": due_at.isoformat(),
        "interval_days": interval,
        "repetitions": reps,
    }


def _update_srs_safe(client: Client, user: AuthUser, *, flashcard_id: str, knew: bool) -> dict[str, Any]:
    def _write() -> dict[str, Any]:
        return update_srs_after_review(client, user, flashcard_id=flashcard_id, knew=knew)

    result = run_or_none_phase14(_write)
    return result or {"flashcard_id": flashcard_id, "due_at": None, "interval_days": 1, "repetitions": 0}


def get_due_flashcard_ids(
    client: Client,
    user: AuthUser,
    *,
    flashcard_ids: list[str],
) -> dict[str, Any]:
    if not flashcard_ids:
        return {"due_count": 0, "due_ids": [], "states": {}}

    now_iso = _now().isoformat()
    states = (
        client.table("flashcard_srs_states")
        .select("flashcard_id, due_at, interval_days, repetitions")
        .eq("user_id", user.id)
        .in_("flashcard_id", flashcard_ids)
        .execute()
        .data
        or []
    )
    state_map = {row["flashcard_id"]: row for row in states}
    due_ids: list[str] = []
    for card_id in flashcard_ids:
        state = state_map.get(card_id)
        if state is None or state.get("due_at", now_iso) <= now_iso:
            due_ids.append(card_id)
    return {
        "due_count": len(due_ids),
        "due_ids": due_ids,
        "states": state_map,
    }
