"""Study session plan — orchestrates brief, flashcards, and quiz for one document."""

from __future__ import annotations

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.services.mastery_service import get_weak_concepts
from app.services.workspace_access import get_accessible_document


async def get_study_session_plan(
    client: Client,
    document_id: str,
    user: AuthUser,
) -> dict[str, Any]:
    doc = get_accessible_document(client, document_id, user, min_role="viewer")
    if doc["file_type"] != "document":
        raise FileException("Study sessions are only available for document uploads")
    if doc["status"] != "ready":
        raise FileException("Document must be processed before starting a study session", status_code=409)

    has_summary = bool(doc.get("summary"))
    flashcards = (
        client.table("flashcard_sets")
        .select("id, title")
        .eq("document_id", document_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )
    quiz = (
        client.table("quizzes")
        .select("id, title, published")
        .eq("document_id", document_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )

    weak_concepts: list[dict[str, Any]] = []
    try:
        weak_concepts = await get_weak_concepts(client, document_id, user)
    except Exception:
        pass

    steps: list[dict[str, Any]] = [
        {
            "step": "brief",
            "label": "Read the brief",
            "ready": has_summary,
            "duration_min": 3,
        },
        {
            "step": "flashcards",
            "label": "Review flashcards",
            "ready": bool(flashcards),
            "set_id": flashcards[0]["id"] if flashcards else None,
            "duration_min": 5,
            "generate_if_missing": True,
        },
        {
            "step": "quiz",
            "label": "Take a quiz",
            "ready": bool(quiz),
            "quiz_id": quiz[0]["id"] if quiz else None,
            "published": quiz[0].get("published") if quiz else False,
            "duration_min": 5,
            "generate_if_missing": True,
        },
    ]

    focus = None
    if weak_concepts:
        focus = weak_concepts[0].get("label") or weak_concepts[0].get("concept_id")
    return {
        "document_id": document_id,
        "filename": doc["filename"],
        "steps": steps,
        "weak_concepts": weak_concepts[:3],
        "focus_topic": focus,
        "estimated_minutes": sum(step["duration_min"] for step in steps),
    }
