"""Rich progress aggregates for study sheet dashboards."""

from __future__ import annotations

from typing import Any

from supabase import Client

from app.core.auth import AuthUser
from app.services.mastery_service import get_workspace_concept_mastery, get_workspace_weak_concepts
from app.services.workspace_access import require_workspace_role
from app.services.workspace_service import get_workspace_stats


async def get_workspace_progress(client: Client, workspace_id: str, user: AuthUser) -> dict[str, Any]:
    stats = get_workspace_stats(client, workspace_id, user)
    require_workspace_role(client, workspace_id, user, min_role="viewer")

    flashcard_reviews = 0
    doc_ids = (
        client.table("documents")
        .select("id")
        .eq("workspace_id", workspace_id)
        .eq("file_type", "document")
        .execute()
        .data
        or []
    )
    document_ids = [row["id"] for row in doc_ids]
    if document_ids:
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
                    .select("id")
                    .eq("user_id", user.id)
                    .in_("flashcard_id", card_ids)
                    .execute()
                    .data
                    or []
                )
                flashcard_reviews = len(reviews)

    weak_concepts: list[dict[str, Any]] = []
    mastery_avg: float | None = None
    try:
        weak_concepts = await get_workspace_weak_concepts(client, workspace_id, user)
        mastery_payload = await get_workspace_concept_mastery(client, workspace_id, user)
        concepts = mastery_payload.get("concepts") or []
        if concepts:
            percents = [float(c.get("percent") or 0) for c in concepts if c.get("attempts", 0) > 0]
            if percents:
                mastery_avg = round(sum(percents) / len(percents), 1)
    except Exception:
        pass

    study_next = _recommend_next(stats, weak_concepts, flashcard_reviews)

    return {
        **stats,
        "flashcard_reviews": flashcard_reviews,
        "mastery_avg_percent": mastery_avg,
        "weak_concepts": weak_concepts[:5],
        "study_next": study_next,
    }


def _recommend_next(
    stats: dict[str, Any],
    weak_concepts: list[dict[str, Any]],
    flashcard_reviews: int,
) -> dict[str, str]:
    if stats.get("ready_count", 0) == 0:
        return {"action": "upload", "label": "Upload and process your first file"}
    if weak_concepts:
        topic = weak_concepts[0].get("name") or weak_concepts[0].get("concept_id") or "weak topics"
        return {"action": "adaptive_quiz", "label": f"Practice {topic}"}
    if stats.get("quiz_attempts", 0) == 0:
        return {"action": "quiz", "label": "Take your first quiz"}
    if flashcard_reviews < 5:
        return {"action": "flashcards", "label": "Review flashcards"}
    return {"action": "course_pack", "label": "Generate a course pack for all sources"}
