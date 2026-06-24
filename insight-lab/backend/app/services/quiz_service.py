import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.base import AppException
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import cache_get, cache_set, check_rate_limit
from app.core.exceptions import NotFoundException, RateLimitException
from app.core.yaml_config import get_yaml_config
from app.core.migration_guard import run_or_raise_phase2
from app.services.llm_client import (
    adaptive_quiz_cache_key,
    excel_quiz_cache_key,
    generate_excel_quiz_draft,
    generate_quiz_draft,
    quiz_cache_key,
    workspace_adaptive_quiz_cache_key,
)
from app.services.mastery_service import get_weak_concepts, record_quiz_mastery
from app.services.quiz_questions import draft_to_rows, excel_draft_to_rows, parse_quiz_draft, QuizQuestionDraft
from app.services.study_session_progress_service import complete_study_session_step_for_quiz
from app.services.workspace_access import (
    can_edit_document,
    get_accessible_document,
    require_editable_document,
)

log = get_logger("quiz")


def _get_owned_document(client: Client, document_id: str, user: AuthUser) -> dict:
    return get_accessible_document(client, document_id, user, min_role="viewer")


def _get_owned_quiz(client: Client, quiz_id: str, user: AuthUser) -> dict:
    result = client.table("quizzes").select("*").eq("id", quiz_id).limit(1).execute()
    if not result.data:
        raise NotFoundException("Quiz not found")
    quiz = result.data[0]
    _get_owned_document(client, quiz["document_id"], user)
    return quiz


def _sample_chunks(chunks: list[dict], max_chunks: int) -> list[dict]:
    if len(chunks) <= max_chunks:
        return chunks
    step = len(chunks) / max_chunks
    indices = sorted({min(int(index * step), len(chunks) - 1) for index in range(max_chunks)})
    return [chunks[index] for index in indices]


def _settings_hash(*, question_type: str, difficulty: str, num_questions: int) -> str:
    raw = f"{question_type}:{difficulty}:{num_questions}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _serialize_question_for_client(row: dict[str, Any], *, include_answers: bool) -> dict[str, Any]:
    payload = {
        "id": row["id"],
        "question_text": row["question_text"],
        "options": row["options"],
        "sort_order": row["sort_order"],
    }
    if include_answers:
        payload["correct_option_index"] = row["correct_option_index"]
        payload["explanation"] = row.get("explanation")
        payload["source_chunk_id"] = row.get("source_chunk_id")
    return payload


def _serialize_quiz(
    quiz: dict[str, Any],
    questions: list[dict[str, Any]],
    *,
    include_answers: bool,
    cached: bool = False,
) -> dict[str, Any]:
    return {
        "quiz_id": quiz["id"],
        "document_id": quiz["document_id"],
        "title": quiz["title"],
        "question_type": quiz["question_type"],
        "difficulty": quiz["difficulty"],
        "published": quiz.get("published", True),
        "questions": [
            _serialize_question_for_client(question, include_answers=include_answers)
            for question in sorted(questions, key=lambda row: row["sort_order"])
        ],
        "cached": cached,
    }


async def generate_document_quiz(
    client: Client,
    document_id: str,
    user: AuthUser,
    *,
    question_type: str,
    difficulty: str,
    num_questions: int,
) -> dict[str, Any]:
    cfg = get_yaml_config().quizzes
    allowed, retry_after = await check_rate_limit(
        key=f"quiz:{user.id}",
        limit=cfg.generate_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Quiz generation limit reached ({cfg.generate_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    num_questions = min(max(num_questions, 1), cfg.max_questions)
    settings = _settings_hash(
        question_type=question_type,
        difficulty=difficulty,
        num_questions=num_questions,
    )
    cache_key = quiz_cache_key(user.id, document_id, settings)
    cached = await cache_get(cache_key)
    if cached and cached.get("quiz_id"):
        quiz = _get_owned_quiz(client, cached["quiz_id"], user)
        questions = (
            client.table("quiz_questions")
            .select("*")
            .eq("quiz_id", quiz["id"])
            .order("sort_order")
            .execute()
            .data
            or []
        )
        return _serialize_quiz(quiz, questions, include_answers=False, cached=True)

    doc = require_editable_document(client, document_id, user)
    if doc["file_type"] != "document":
        raise FileException("Quizzes are only available for document uploads")
    if doc["status"] != "ready":
        raise FileException("Document must be processed before generating a quiz", status_code=409)

    chunks_result = (
        client.table("document_chunks")
        .select("chunk_index, content")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
    )
    chunks = chunks_result.data or []
    if not chunks:
        raise FileException("No document chunks found; reprocess the document", status_code=409)

    sampled = _sample_chunks(chunks, cfg.max_context_chunks)
    chunk_indexes = [row["chunk_index"] for row in sampled]
    context_chunks = [row["content"] for row in sampled]

    raw = await generate_quiz_draft(
        context_chunks=context_chunks,
        filename=doc["filename"],
        question_type=question_type,
        difficulty=difficulty,
        num_questions=num_questions,
    )
    try:
        draft = parse_quiz_draft(raw, max_questions=num_questions)
    except (ValueError, Exception) as exc:
        raise FileException(f"Invalid quiz payload from LLM: {exc}", status_code=502) from exc

    quiz_id = str(uuid4())
    question_rows = draft_to_rows(draft, chunk_indexes=chunk_indexes)
    for row in question_rows:
        row["id"] = str(uuid4())
        row["quiz_id"] = quiz_id

    quiz_row = {
        "id": quiz_id,
        "document_id": document_id,
        "workspace_id": doc["workspace_id"],
        "title": draft.title.strip() or f"Quiz: {doc['filename']}",
        "question_type": question_type,
        "difficulty": difficulty,
        "published": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    client.table("quizzes").insert(quiz_row).execute()
    client.table("quiz_questions").insert(question_rows).execute()

    payload = _serialize_quiz(quiz_row, question_rows, include_answers=False, cached=False)
    await cache_set(cache_key, {"quiz_id": quiz_id}, get_yaml_config().cache.quiz_ttl)
    log.info(
        "Quiz generated",
        quiz_id=quiz_id,
        document_id=document_id,
        user_id=user.id,
        question_count=len(question_rows),
    )
    return payload


def _weak_concepts_hash(concept_ids: list[str]) -> str:
    joined = ",".join(sorted(concept_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _chunks_for_concepts(
    client: Client,
    *,
    document_id: str,
    concept_ids: list[str],
    max_chunks: int,
) -> tuple[list[dict], list[int]]:
    if not concept_ids:
        return [], []

    concepts = run_or_raise_phase2(
        lambda: client.table("document_concepts")
        .select("concept_id, chunk_indexes")
        .eq("document_id", document_id)
        .in_("concept_id", concept_ids)
        .execute()
        .data
        or []
    )
    target_indexes: set[int] = set()
    for concept in concepts:
        for index in concept.get("chunk_indexes") or []:
            target_indexes.add(int(index))

    chunks_result = (
        client.table("document_chunks")
        .select("chunk_index, content")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
    )
    all_chunks = chunks_result.data or []
    if target_indexes:
        selected = [row for row in all_chunks if row["chunk_index"] in target_indexes]
    else:
        selected = _sample_chunks(all_chunks, max_chunks)
    selected = selected[:max_chunks]
    return selected, [row["chunk_index"] for row in selected]


async def generate_adaptive_quiz(
    client: Client,
    document_id: str,
    user: AuthUser,
    *,
    question_type: str,
    difficulty: str,
    num_questions: int,
) -> dict[str, Any]:
    cfg = get_yaml_config().adaptive_quiz
    allowed, retry_after = await check_rate_limit(
        key=f"adaptive_quiz:{user.id}",
        limit=cfg.generate_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Adaptive quiz limit reached ({cfg.generate_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    weak = await get_weak_concepts(client, document_id, user)
    if not weak:
        raise FileException(
            "Complete a quiz first. We'll then suggest topics to practice.",
            status_code=409,
        )

    weak_ids = [row["concept_id"] for row in weak]
    weak_names = [row["name"] for row in weak]
    cache_key = adaptive_quiz_cache_key(user.id, document_id, _weak_concepts_hash(weak_ids))
    cached = await cache_get(cache_key)
    if cached and cached.get("quiz_id"):
        quiz = _get_owned_quiz(client, cached["quiz_id"], user)
        questions = (
            client.table("quiz_questions")
            .select("*")
            .eq("quiz_id", quiz["id"])
            .order("sort_order")
            .execute()
            .data
            or []
        )
        return {
            **_serialize_quiz(quiz, questions, include_answers=False, cached=True),
            "target_concepts": weak,
        }

    doc = require_editable_document(client, document_id, user)
    if doc["file_type"] != "document":
        raise FileException("Quizzes are only available for document uploads")
    if doc["status"] != "ready":
        raise FileException("Document must be processed before generating a quiz", status_code=409)

    quiz_cfg = get_yaml_config().quizzes
    num_questions = min(max(num_questions, 1), quiz_cfg.max_questions)
    sampled, chunk_indexes = _chunks_for_concepts(
        client,
        document_id=document_id,
        concept_ids=weak_ids,
        max_chunks=quiz_cfg.max_context_chunks,
    )
    if not sampled:
        raise FileException("No chunks found for weak concepts; sync the concept graph first", status_code=409)

    context_chunks = [row["content"] for row in sampled]
    raw = await generate_quiz_draft(
        context_chunks=context_chunks,
        filename=doc["filename"],
        question_type=question_type,
        difficulty=difficulty,
        num_questions=num_questions,
        focus_concepts=weak_names,
    )
    try:
        draft = parse_quiz_draft(raw, max_questions=num_questions)
    except (ValueError, Exception) as exc:
        raise FileException(f"Invalid quiz payload from LLM: {exc}", status_code=502) from exc

    quiz_id = str(uuid4())
    question_rows = draft_to_rows(draft, chunk_indexes=chunk_indexes)
    for index, row in enumerate(question_rows):
        row["id"] = str(uuid4())
        row["quiz_id"] = quiz_id
        if not row.get("concept_id") and index < len(weak_ids):
            row["concept_id"] = weak_ids[index % len(weak_ids)]

    quiz_row = {
        "id": quiz_id,
        "document_id": document_id,
        "workspace_id": doc["workspace_id"],
        "title": draft.title.strip() or f"Adaptive quiz: {doc['filename']}",
        "question_type": question_type,
        "difficulty": difficulty,
        "published": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    client.table("quizzes").insert(quiz_row).execute()
    client.table("quiz_questions").insert(question_rows).execute()

    payload = _serialize_quiz(quiz_row, question_rows, include_answers=False, cached=False)
    await cache_set(cache_key, {"quiz_id": quiz_id}, get_yaml_config().cache.quiz_ttl)
    log.info(
        "Adaptive quiz generated",
        quiz_id=quiz_id,
        document_id=document_id,
        user_id=user.id,
        weak_concepts=weak_ids,
    )
    return {**payload, "target_concepts": weak}


async def generate_workspace_adaptive_quiz(
    client: Client,
    workspace_id: str,
    user: AuthUser,
    *,
    question_type: str,
    difficulty: str,
    num_questions: int,
) -> dict[str, Any]:
    from app.services.mastery_service import get_workspace_weak_concepts
    from app.services.workspace_access import require_workspace_role

    cfg = get_yaml_config().adaptive_quiz
    allowed, retry_after = await check_rate_limit(
        key=f"workspace_adaptive_quiz:{user.id}",
        limit=cfg.generate_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Adaptive quiz limit reached ({cfg.generate_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    require_workspace_role(client, workspace_id, user, min_role="editor")
    weak = await get_workspace_weak_concepts(client, workspace_id, user)
    if not weak:
        raise FileException(
            "Complete a quiz on at least one document in this set first. "
            "We'll then suggest topics to practice across the set.",
            status_code=409,
        )

    weak_ids = [row["concept_id"] for row in weak]
    weak_names = [row["name"] for row in weak]
    cache_key = workspace_adaptive_quiz_cache_key(
        user.id,
        workspace_id,
        _weak_concepts_hash(weak_ids),
    )
    cached = await cache_get(cache_key)
    if cached and cached.get("quiz_id"):
        quiz = _get_owned_quiz(client, cached["quiz_id"], user)
        questions = (
            client.table("quiz_questions")
            .select("*")
            .eq("quiz_id", quiz["id"])
            .order("sort_order")
            .execute()
            .data
            or []
        )
        return {
            **_serialize_quiz(quiz, questions, include_answers=False, cached=True),
            "workspace_id": workspace_id,
            "target_concepts": weak,
        }

    quiz_cfg = get_yaml_config().quizzes
    num_questions = min(max(num_questions, 1), quiz_cfg.max_questions)
    context_chunks: list[str] = []
    chunk_indexes: list[int] = []
    primary_document_id: str | None = None
    chunks_per_concept = max(1, quiz_cfg.max_context_chunks // max(len(weak), 1))

    for concept in weak:
        document_id = concept["document_id"]
        sampled, indexes = _chunks_for_concepts(
            client,
            document_id=document_id,
            concept_ids=[concept["concept_id"]],
            max_chunks=chunks_per_concept,
        )
        if not sampled:
            continue
        if primary_document_id is None:
            primary_document_id = document_id
        context_chunks.extend(row["content"] for row in sampled)
        chunk_indexes.extend(indexes)
        if len(context_chunks) >= quiz_cfg.max_context_chunks:
            break

    if not context_chunks or not primary_document_id:
        raise FileException(
            "No chunks found for weak concepts; sync concept graphs for documents in this set",
            status_code=409,
        )

    context_chunks = context_chunks[: quiz_cfg.max_context_chunks]
    primary_doc = _get_owned_document(client, primary_document_id, user)
    raw = await generate_quiz_draft(
        context_chunks=context_chunks,
        filename=primary_doc["filename"],
        question_type=question_type,
        difficulty=difficulty,
        num_questions=num_questions,
        focus_concepts=weak_names,
    )
    try:
        draft = parse_quiz_draft(raw, max_questions=num_questions)
    except (ValueError, Exception) as exc:
        raise FileException(f"Invalid quiz payload from LLM: {exc}", status_code=502) from exc

    quiz_id = str(uuid4())
    question_rows = draft_to_rows(draft, chunk_indexes=chunk_indexes[: len(draft.questions)])
    for index, row in enumerate(question_rows):
        row["id"] = str(uuid4())
        row["quiz_id"] = quiz_id
        if not row.get("concept_id") and index < len(weak_ids):
            row["concept_id"] = weak_ids[index % len(weak_ids)]

    quiz_row = {
        "id": quiz_id,
        "document_id": primary_document_id,
        "workspace_id": workspace_id,
        "title": draft.title.strip() or "Adaptive quiz: study set",
        "question_type": question_type,
        "difficulty": difficulty,
        "published": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    client.table("quizzes").insert(quiz_row).execute()
    client.table("quiz_questions").insert(question_rows).execute()

    payload = _serialize_quiz(quiz_row, question_rows, include_answers=False, cached=False)
    await cache_set(cache_key, {"quiz_id": quiz_id}, get_yaml_config().cache.quiz_ttl)
    log.info(
        "Workspace adaptive quiz generated",
        quiz_id=quiz_id,
        workspace_id=workspace_id,
        user_id=user.id,
        weak_concepts=weak_ids,
    )
    return {
        **payload,
        "workspace_id": workspace_id,
        "target_concepts": weak,
    }


async def generate_excel_quiz(
    client: Client,
    document_id: str,
    user: AuthUser,
    *,
    question_type: str,
    difficulty: str,
    num_questions: int,
) -> dict[str, Any]:
    cfg = get_yaml_config().quizzes
    allowed, retry_after = await check_rate_limit(
        key=f"excel_quiz:{user.id}",
        limit=cfg.generate_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Quiz generation limit reached ({cfg.generate_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    num_questions = min(max(num_questions, 1), cfg.max_questions)
    settings = _settings_hash(
        question_type=question_type,
        difficulty=difficulty,
        num_questions=num_questions,
    )
    cache_key = excel_quiz_cache_key(user.id, document_id, settings)
    cached = await cache_get(cache_key)
    if cached and cached.get("quiz_id"):
        quiz = _get_owned_quiz(client, cached["quiz_id"], user)
        questions = (
            client.table("quiz_questions")
            .select("*")
            .eq("quiz_id", quiz["id"])
            .order("sort_order")
            .execute()
            .data
            or []
        )
        return _serialize_quiz(quiz, questions, include_answers=False, cached=True)

    doc = require_editable_document(client, document_id, user)
    if doc["file_type"] != "excel":
        raise FileException("Excel quizzes are only available for spreadsheet uploads")
    if doc["status"] != "ready" or not doc.get("excel_summary"):
        raise FileException("Spreadsheet must be analyzed before generating a quiz", status_code=409)

    profile = doc.get("excel_profile") or {}
    summary = doc.get("excel_summary") or ""
    charts = doc.get("excel_charts") or []
    if not profile:
        raise FileException("Spreadsheet profile missing — re-analyze the file", status_code=409)

    raw = await generate_excel_quiz_draft(
        profile=profile,
        summary=summary,
        charts=charts,
        filename=doc["filename"],
        question_type=question_type,
        difficulty=difficulty,
        num_questions=num_questions,
    )
    try:
        draft = parse_quiz_draft(raw, max_questions=num_questions)
    except (ValueError, Exception) as exc:
        raise FileException(f"Invalid quiz payload from LLM: {exc}", status_code=502) from exc

    chart_ids = [str(chart.get("id") or index) for index, chart in enumerate(charts)]
    quiz_id = str(uuid4())
    question_rows = excel_draft_to_rows(draft, chart_ids=chart_ids)
    for row in question_rows:
        row["id"] = str(uuid4())
        row["quiz_id"] = quiz_id

    quiz_row = {
        "id": quiz_id,
        "document_id": document_id,
        "workspace_id": doc["workspace_id"],
        "title": draft.title.strip() or f"Quiz: {doc['filename']}",
        "question_type": question_type,
        "difficulty": difficulty,
        "published": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    client.table("quizzes").insert(quiz_row).execute()
    client.table("quiz_questions").insert(question_rows).execute()

    payload = _serialize_quiz(quiz_row, question_rows, include_answers=False, cached=False)
    await cache_set(cache_key, {"quiz_id": quiz_id}, get_yaml_config().cache.quiz_ttl)
    log.info(
        "Excel quiz generated",
        quiz_id=quiz_id,
        document_id=document_id,
        user_id=user.id,
        question_count=len(question_rows),
    )
    return payload


async def get_document_quiz(client: Client, document_id: str, user: AuthUser) -> dict[str, Any] | None:
    _get_owned_document(client, document_id, user)
    quiz_result = (
        client.table("quizzes")
        .select("*")
        .eq("document_id", document_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not quiz_result.data:
        return None

    quiz = quiz_result.data[0]
    questions = (
        client.table("quiz_questions")
        .select("*")
        .eq("quiz_id", quiz["id"])
        .order("sort_order")
        .execute()
        .data
        or []
    )
    return _serialize_quiz(quiz, questions, include_answers=False, cached=False)


async def update_quiz_question(
    client: Client,
    quiz_id: str,
    question_id: str,
    user: AuthUser,
    *,
    question_text: str | None = None,
    options: list[str] | None = None,
    correct_option_index: int | None = None,
    explanation: str | None = None,
) -> dict[str, Any]:
    quiz = _get_owned_quiz(client, quiz_id, user)
    if not can_edit_document(client, quiz["document_id"], user):
        raise FileException("You do not have permission to edit this quiz", status_code=403)

    question_result = (
        client.table("quiz_questions")
        .select("*")
        .eq("id", question_id)
        .eq("quiz_id", quiz_id)
        .limit(1)
        .execute()
    )
    if not question_result.data:
        raise NotFoundException("Quiz question not found")
    current = question_result.data[0]

    draft = QuizQuestionDraft(
        question_text=question_text if question_text is not None else current["question_text"],
        options=options if options is not None else current["options"],
        correct_option_index=(
            correct_option_index
            if correct_option_index is not None
            else current["correct_option_index"]
        ),
        explanation=explanation if explanation is not None else current.get("explanation"),
    )
    updated = (
        client.table("quiz_questions")
        .update(
            {
                "question_text": draft.question_text,
                "options": draft.options,
                "correct_option_index": draft.correct_option_index,
                "explanation": draft.explanation,
            }
        )
        .eq("id", question_id)
        .eq("quiz_id", quiz_id)
        .execute()
    )
    if not updated.data:
        raise FileException("Failed to update quiz question", status_code=500)

    client.table("quizzes").update(
        {"published": False, "updated_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", quiz_id).execute()

    row = updated.data[0]
    return _serialize_question_for_client(row, include_answers=True)


async def publish_quiz(client: Client, quiz_id: str, user: AuthUser) -> dict[str, Any]:
    from app.services.source_links_service import new_public_share_token

    quiz = _get_owned_quiz(client, quiz_id, user)
    if not can_edit_document(client, quiz["document_id"], user):
        raise FileException("You do not have permission to publish this quiz", status_code=403)

    share_token = quiz.get("public_share_token") or new_public_share_token()
    updated = (
        client.table("quizzes")
        .update(
            {
                "published": True,
                "public_share_token": share_token,
                "public_share_enabled_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .eq("id", quiz_id)
        .execute()
    )
    if not updated.data:
        raise FileException("Failed to publish quiz", status_code=500)
    row = updated.data[0]
    questions = (
        client.table("quiz_questions")
        .select("*")
        .eq("quiz_id", quiz_id)
        .order("sort_order")
        .execute()
        .data
        or []
    )
    serialized = _serialize_quiz(row, questions, include_answers=False, cached=False)
    serialized["public_share_token"] = share_token
    return serialized


async def _load_public_quiz(client: Client, *, share_token: str) -> dict[str, Any]:
    result = (
        client.table("quizzes")
        .select("*")
        .eq("public_share_token", share_token.strip())
        .eq("published", True)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Quiz not found or not published")
    quiz = result.data[0]
    questions = (
        client.table("quiz_questions")
        .select("*")
        .eq("quiz_id", quiz["id"])
        .order("sort_order")
        .execute()
        .data
        or []
    )
    doc = (
        client.table("documents")
        .select("filename")
        .eq("id", quiz["document_id"])
        .limit(1)
        .execute()
    )
    serialized = _serialize_quiz(quiz, questions, include_answers=False, cached=False)
    serialized["source_filename"] = doc.data[0]["filename"] if doc.data else None
    return serialized


async def get_public_quiz(client: Client, *, share_token: str, client_ip: str = "unknown") -> dict[str, Any]:
    quiz_cfg = get_yaml_config().quizzes
    token_key = share_token.strip()[:24] or "unknown"
    allowed, retry_after = await check_rate_limit(
        key=f"public_quiz:get:{client_ip}:{token_key}",
        limit=quiz_cfg.public_get_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Too many requests ({quiz_cfg.public_get_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    return await _load_public_quiz(client, share_token=share_token)


async def submit_public_quiz(
    client: Client,
    *,
    share_token: str,
    display_name: str,
    answers: dict[str, int],
    client_ip: str = "unknown",
) -> dict[str, Any]:
    quiz_cfg = get_yaml_config().quizzes
    token_key = share_token.strip()[:24] or "unknown"
    allowed, retry_after = await check_rate_limit(
        key=f"public_quiz:submit:{client_ip}:{token_key}",
        limit=quiz_cfg.public_submit_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Too many submissions ({quiz_cfg.public_submit_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    import json

    if len(json.dumps(answers)) > quiz_cfg.public_max_answers_bytes:
        raise FileException("Answers payload too large", status_code=400)

    quiz_payload = await _load_public_quiz(client, share_token=share_token)
    quiz_id = quiz_payload["quiz_id"]
    questions = (
        client.table("quiz_questions")
        .select("*")
        .eq("quiz_id", quiz_id)
        .order("sort_order")
        .execute()
        .data
        or []
    )
    if not questions:
        raise FileException("Quiz has no questions", status_code=409)

    if len(answers) > quiz_cfg.max_questions:
        raise FileException("Too many answers submitted", status_code=400)

    safe_name = (display_name or "Guest").strip()[:80] or "Guest"
    score = 0
    results: list[dict[str, Any]] = []
    normalized_answers: dict[str, int] = {}
    for question in questions:
        question_id = question["id"]
        selected = answers.get(question_id)
        if selected is None:
            raise FileException(f"Missing answer for question {question_id}", status_code=400)
        if not isinstance(selected, int) or selected < 0 or selected >= len(question["options"]):
            raise FileException(f"Invalid answer for question {question_id}", status_code=400)
        normalized_answers[question_id] = selected
        correct_index = question["correct_option_index"]
        is_correct = selected == correct_index
        if is_correct:
            score += 1
        results.append(
            {
                "question_id": question_id,
                "correct": is_correct,
                "selected_option_index": selected,
                "correct_option_index": correct_index,
                "explanation": question.get("explanation"),
            }
        )

    total = len(questions)
    client.table("quiz_public_attempts").insert(
        {
            "quiz_id": quiz_id,
            "display_name": safe_name,
            "score": score,
            "total": total,
            "answers": normalized_answers,
        }
    ).execute()

    return {
        "quiz_id": quiz_id,
        "display_name": safe_name,
        "score": score,
        "total": total,
        "percent": round((score / total) * 100, 1) if total else 0,
        "results": results,
    }


async def get_quiz_for_edit(client: Client, quiz_id: str, user: AuthUser) -> dict[str, Any]:
    quiz = _get_owned_quiz(client, quiz_id, user)
    if not can_edit_document(client, quiz["document_id"], user):
        raise FileException("You do not have permission to edit this quiz", status_code=403)
    questions = (
        client.table("quiz_questions")
        .select("*")
        .eq("quiz_id", quiz_id)
        .order("sort_order")
        .execute()
        .data
        or []
    )
    return _serialize_quiz(quiz, questions, include_answers=True, cached=False)


async def submit_quiz_attempt(
    client: Client,
    quiz_id: str,
    user: AuthUser,
    *,
    answers: dict[str, int],
    study_session_step_id: str | None = None,
) -> dict[str, Any]:
    quiz_cfg = get_yaml_config().quizzes
    allowed, retry_after = await check_rate_limit(
        key=f"quiz_submit:{user.id}",
        limit=quiz_cfg.submit_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Quiz submit limit reached ({quiz_cfg.submit_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    quiz = _get_owned_quiz(client, quiz_id, user)
    questions = (
        client.table("quiz_questions")
        .select("*")
        .eq("quiz_id", quiz_id)
        .order("sort_order")
        .execute()
        .data
        or []
    )
    if not questions:
        raise FileException("Quiz has no questions", status_code=409)

    if len(answers) > quiz_cfg.max_questions:
        raise FileException("Too many answers submitted", status_code=400)

    results: list[dict[str, Any]] = []
    score = 0
    normalized_answers: dict[str, int] = {}
    for question in questions:
        question_id = question["id"]
        selected = answers.get(question_id)
        if selected is None:
            raise FileException(f"Missing answer for question {question_id}", status_code=400)
        if not isinstance(selected, int) or selected < 0 or selected >= len(question["options"]):
            raise FileException(f"Invalid answer for question {question_id}", status_code=400)
        normalized_answers[question_id] = selected
        correct = selected == question["correct_option_index"]
        if correct:
            score += 1
        results.append(
            {
                "question_id": question_id,
                "question_text": question["question_text"],
                "selected_option_index": selected,
                "correct_option_index": question["correct_option_index"],
                "correct": correct,
                "explanation": question.get("explanation"),
            }
        )

    total = len(questions)
    attempt = (
        client.table("quiz_attempts")
        .insert(
            {
                "quiz_id": quiz_id,
                "user_id": user.id,
                "score": score,
                "total": total,
                "answers": normalized_answers,
            }
        )
        .execute()
    )
    attempt_id = attempt.data[0]["id"] if attempt.data else None
    record_quiz_mastery(
        client,
        user=user,
        document_id=quiz["document_id"],
        questions=questions,
        results=results,
    )
    log.info("Quiz submitted", quiz_id=quiz_id, user_id=user.id, score=score, total=total)
    complete_study_session_step_for_quiz(
        client,
        user=user,
        study_session_step_id=study_session_step_id,
        attempt_id=attempt_id,
    )
    from app.services.document_extras import attach_source_previews

    enriched_results = attach_source_previews(
        client,
        document_id=quiz["document_id"],
        results=results,
        questions=questions,
    )
    return {
        "attempt_id": attempt_id,
        "quiz_id": quiz_id,
        "document_id": quiz["document_id"],
        "score": score,
        "total": total,
        "percent": round((score / total) * 100, 1) if total else 0.0,
        "results": enriched_results,
    }
