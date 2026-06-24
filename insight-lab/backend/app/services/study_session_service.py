"""Study session plans — document-level and workspace-level guided study flows."""

from __future__ import annotations

from typing import Any

from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.services.mastery_service import get_weak_concepts, get_workspace_weak_concepts
from app.services.workspace_access import get_accessible_document, require_workspace_role


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
        focus = weak_concepts[0].get("name") or weak_concepts[0].get("concept_id")
    return {
        "document_id": document_id,
        "filename": doc["filename"],
        "steps": steps,
        "weak_concepts": weak_concepts[:3],
        "focus_topic": focus,
        "estimated_minutes": sum(step["duration_min"] for step in steps),
    }


async def get_workspace_study_session_plan(
    client: Client,
    workspace_id: str,
    user: AuthUser,
) -> dict[str, Any]:
    require_workspace_role(client, workspace_id, user, min_role="viewer")

    docs = (
        client.table("documents")
        .select("id, filename, summary")
        .eq("workspace_id", workspace_id)
        .eq("file_type", "document")
        .eq("status", "ready")
        .order("created_at")
        .execute()
        .data
        or []
    )
    if not docs:
        raise FileException(
            "No ready documents in this study set — upload and process PDFs or Word files first",
            status_code=409,
        )

    weak_concepts: list[dict[str, Any]] = []
    try:
        weak_concepts = await get_workspace_weak_concepts(client, workspace_id, user)
    except Exception:
        pass

    focus = None
    if weak_concepts:
        focus = weak_concepts[0].get("name") or weak_concepts[0].get("concept_id")

    steps: list[dict[str, Any]] = [
        {
            "step": "focus",
            "label": "Review focus topics",
            "ready": True,
            "duration_min": 2,
            "weak_concepts": weak_concepts[:5],
            "focus_topic": focus,
        },
    ]

    for doc in docs:
        steps.append(
            {
                "step": "brief",
                "label": f"Read brief — {doc['filename']}",
                "ready": bool(doc.get("summary")),
                "duration_min": 3,
                "document_id": doc["id"],
                "filename": doc["filename"],
            }
        )

    for doc in docs:
        flashcards = (
            client.table("flashcard_sets")
            .select("id")
            .eq("document_id", doc["id"])
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        steps.append(
            {
                "step": "flashcards",
                "label": f"Flashcards — {doc['filename']}",
                "ready": bool(flashcards),
                "duration_min": 5,
                "document_id": doc["id"],
                "filename": doc["filename"],
                "set_id": flashcards[0]["id"] if flashcards else None,
                "generate_if_missing": True,
            }
        )

    if weak_concepts:
        steps.append(
            {
                "step": "adaptive_quiz",
                "label": "Practice weak areas",
                "ready": True,
                "duration_min": 8,
                "weak_count": len(weak_concepts),
            }
        )
    else:
        steps.append(
            {
                "step": "set_quiz",
                "label": "Take a set-wide quiz",
                "ready": True,
                "duration_min": 8,
                "hint": "Complete quizzes to unlock adaptive practice on weak topics",
            }
        )

    return {
        "workspace_id": workspace_id,
        "document_count": len(docs),
        "steps": steps,
        "weak_concepts": weak_concepts[:5],
        "focus_topic": focus,
        "estimated_minutes": sum(step["duration_min"] for step in steps),
    }
