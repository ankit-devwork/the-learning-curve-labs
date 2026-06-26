import asyncio
from datetime import datetime, timezone

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.base import AppException
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import cache_get, cache_set, check_rate_limit
from app.core.exceptions import ForbiddenException, NotFoundException, RateLimitException
from app.core.safe_errors import GENERIC_EXCEL_ERROR
from app.core.resilience import storage_circuit, with_retry
from app.core.yaml_config import get_yaml_config
from app.services.excel_charts import (
    CustomChartRequest,
    build_charts_from_plan,
    build_custom_chart,
    parse_chart_plan,
)
from app.services.excel_profiling import profile_dataframe, read_dataframe
from app.services.llm_client import (
    answer_excel_question,
    excel_cache_key,
    excel_question_cache_key,
    generate_excel_chart_plan,
    generate_excel_summary,
    semantic_excel_chat_index_key,
)
from app.services.semantic_cache import (
    get_semantic_cached_answer_by_index,
    store_semantic_cached_answer_by_index,
)
from app.services.document_storage import download_document_blob
from app.services.workspace_access import get_accessible_document, require_editable_document
from app.services.source_links_service import (
    build_linked_citations,
    fetch_linked_document_chunks,
)

log = get_logger("excel")


def _get_readable_document(client: Client, document_id: str, user: AuthUser) -> dict:
    return get_accessible_document(client, document_id, user, min_role="viewer")


async def _download_document_bytes(client: Client, document: dict) -> bytes:
    async def _fetch() -> bytes:
        return await asyncio.to_thread(download_document_blob, client, document)

    async def _with_retry() -> bytes:
        return await with_retry(_fetch, operation="storage.download")

    try:
        return await storage_circuit.call(_with_retry)
    except AppException:
        raise
    except Exception as exc:
        raise FileException(f"Failed to download file from storage: {exc}", status_code=502) from exc


def _serialize_result(
    *,
    document_id: str,
    profile: dict,
    charts: list[dict],
    summary: str,
    status: str,
    cached: bool,
) -> dict:
    return {
        "document_id": document_id,
        "status": status,
        "profile": profile,
        "charts": charts,
        "summary": summary,
        "cached": cached,
    }


async def get_excel_analysis(client: Client, document_id: str, user: AuthUser) -> dict:
    doc = _get_readable_document(client, document_id, user)
    if doc["file_type"] != "excel":
        raise FileException("Excel analysis is only available for spreadsheet uploads")

    cache_key = excel_cache_key(user.id, document_id, doc.get("file_hash"))
    cached = await cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    if doc["status"] != "ready" or not doc.get("excel_charts"):
        raise FileException("Spreadsheet is not analyzed yet", status_code=409)

    payload = _serialize_result(
        document_id=document_id,
        profile=doc.get("excel_profile") or {},
        charts=doc.get("excel_charts") or [],
        summary=doc.get("excel_summary") or "",
        status=doc["status"],
        cached=False,
    )
    await cache_set(cache_key, payload, get_yaml_config().cache.excel_ttl)
    return payload


async def analyze_excel(client: Client, document_id: str, user: AuthUser) -> dict:
    cfg = get_yaml_config().excel
    allowed, retry_after = await check_rate_limit(
        key=f"excel:{user.id}",
        limit=cfg.analyze_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Excel analyze rate limit reached ({cfg.analyze_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    doc = require_editable_document(client, document_id, user)
    if doc["file_type"] != "excel":
        raise FileException("Only spreadsheet uploads can be analyzed")
    if doc["status"] == "processing":
        raise FileException("Spreadsheet is already being analyzed", status_code=409)

    cache_key = excel_cache_key(user.id, document_id, doc.get("file_hash"))
    cached = await cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    client.table("documents").update(
        {"status": "processing", "error_message": None}
    ).eq("id", document_id).execute()

    try:
        raw_bytes = await _download_document_bytes(client, doc)
        df = read_dataframe(raw_bytes, doc["filename"])
        profile = profile_dataframe(df)

        plan_raw = await generate_excel_chart_plan(profile=profile, filename=doc["filename"])
        try:
            plan_items = parse_chart_plan(plan_raw, max_charts=cfg.max_charts)
        except (ValueError, Exception) as exc:
            raise FileException(f"Invalid chart plan from LLM: {exc}", status_code=502) from exc
        charts = build_charts_from_plan(df, plan_items)
        if not charts:
            raise FileException("Could not build any charts from the spreadsheet")

        summary = await generate_excel_summary(
            profile=profile,
            charts=charts,
            filename=doc["filename"],
        )

        client.table("documents").update(
            {
                "status": "ready",
                "excel_profile": profile,
                "excel_charts": charts,
                "excel_summary": summary,
                "error_message": None,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", document_id).execute()

        payload = _serialize_result(
            document_id=document_id,
            profile=profile,
            charts=charts,
            summary=summary,
            status="ready",
            cached=False,
        )
        await cache_set(cache_key, payload, get_yaml_config().cache.excel_ttl)
        log.info(
            "Excel analyzed",
            document_id=document_id,
            user_id=user.id,
            chart_count=len(charts),
            row_count=profile.get("row_count"),
        )
        return payload
    except AppException as exc:
        client.table("documents").update(
            {"status": "failed", "error_message": exc.message}
        ).eq("id", document_id).execute()
        raise
    except Exception as exc:
        log.exception("Excel analysis failed", document_id=document_id, user_id=user.id)
        client.table("documents").update(
            {"status": "failed", "error_message": GENERIC_EXCEL_ERROR}
        ).eq("id", document_id).execute()
        raise FileException(GENERIC_EXCEL_ERROR, status_code=500) from exc


async def create_custom_excel_chart(
    client: Client,
    document_id: str,
    user: AuthUser,
    request: CustomChartRequest,
) -> dict:
    doc = require_editable_document(client, document_id, user)
    if doc["file_type"] != "excel":
        raise FileException("Custom charts are only available for spreadsheet uploads")
    if doc["status"] != "ready" or not doc.get("excel_profile"):
        raise FileException("Analyze the spreadsheet before building custom charts", status_code=409)

    raw_bytes = await _download_document_bytes(client, doc)
    df = read_dataframe(raw_bytes, doc["filename"])

    try:
        chart = build_custom_chart(df, request)
    except ValueError as exc:
        raise FileException(str(exc), status_code=400) from exc

    log.info(
        "Custom Excel chart built",
        document_id=document_id,
        user_id=user.id,
        chart_type=request.chart_type,
        x_column=request.x_column,
        y_column=request.y_column,
    )
    return {"chart": chart}


async def ask_excel(
    client: Client,
    document_id: str,
    user: AuthUser,
    *,
    question: str,
) -> dict:
    question = question.strip()
    if not question:
        raise FileException("Question is required")

    cfg = get_yaml_config().excel
    allowed, retry_after = await check_rate_limit(
        key=f"excel_chat:{user.id}",
        limit=cfg.chat_rate_limit_per_min,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitException(
            f"Excel chat rate limit reached ({cfg.chat_rate_limit_per_min}/min)",
            retry_after=retry_after,
        )

    doc = _get_readable_document(client, document_id, user)
    if doc["file_type"] != "excel":
        raise FileException("Excel chat is only available for spreadsheet uploads")
    if doc["status"] != "ready" or not doc.get("excel_profile"):
        raise FileException("Analyze the spreadsheet before asking questions", status_code=409)

    cache_key = excel_question_cache_key(user.id, document_id, doc.get("file_hash"), question)
    cached = await cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    semantic_index_key = semantic_excel_chat_index_key(
        user.id,
        document_id,
        doc.get("file_hash"),
    )
    semantic_cached = await get_semantic_cached_answer_by_index(
        index_key=semantic_index_key,
        question=question,
    )
    if semantic_cached:
        _get_readable_document(client, document_id, user)
        return semantic_cached

    profile = doc.get("excel_profile") or {}
    charts = doc.get("excel_charts") or []
    summary = doc.get("excel_summary") or ""
    linked_citations: list[dict] = []

    try:
        linked_rows = fetch_linked_document_chunks(
            client,
            excel_document_id=document_id,
            question=question,
            limit=3,
        )
        linked_citations = build_linked_citations(linked_rows)
        linked_excerpts = [row["content"] for row in linked_rows]
        llm_result = await answer_excel_question(
            question=question,
            profile=profile,
            summary=summary,
            charts=charts,
            filename=doc["filename"],
            linked_excerpts=linked_excerpts,
        )
    except Exception as exc:
        log.exception("Excel chat failed", document_id=document_id, user_id=user.id)
        raise FileException(GENERIC_EXCEL_ERROR, status_code=500) from exc

    payload = {
        "document_id": document_id,
        "question": question,
        "answer": llm_result["answer"],
        "sources": llm_result["sources"],
        "document_citations": linked_citations,
        "cached": False,
    }
    await cache_set(cache_key, payload, get_yaml_config().cache.chat_ttl)
    await store_semantic_cached_answer_by_index(
        index_key=semantic_index_key,
        question=question,
        payload=payload,
    )
    log.info("Excel chat answered", document_id=document_id, user_id=user.id)
    return payload


async def get_excel_preview(
    client: Client,
    document_id: str,
    user: AuthUser,
    *,
    limit: int | None = None,
) -> dict:
    cfg = get_yaml_config().excel_preview
    row_limit = min(max(limit or cfg.default_limit, 1), cfg.max_limit)

    doc = _get_readable_document(client, document_id, user)
    if doc["file_type"] != "excel":
        raise FileException("Excel preview is only available for spreadsheet uploads")

    raw_bytes = await _download_document_bytes(client, doc)
    df = read_dataframe(raw_bytes, doc["filename"])
    preview = df.head(row_limit).fillna("").astype(str)
    columns = [str(column) for column in preview.columns.tolist()]
    rows = preview.values.tolist()

    return {
        "document_id": document_id,
        "columns": columns,
        "rows": rows,
        "preview_rows": len(rows),
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
    }
