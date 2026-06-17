import asyncio
from datetime import datetime, timezone

from pycorekit.core_logging.logger import get_logger
from pycorekit.exceptions.base import AppException
from pycorekit.exceptions.file import FileException
from supabase import Client

from app.core.auth import AuthUser
from app.core.cache import cache_get, cache_set, check_rate_limit
from app.core.exceptions import NotFoundException, RateLimitException
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
    excel_cache_key,
    generate_excel_chart_plan,
    generate_excel_summary,
)

log = get_logger("excel")


def _get_owned_document(client: Client, document_id: str, user: AuthUser) -> dict:
    result = (
        client.table("documents")
        .select("*")
        .eq("id", document_id)
        .eq("owner_id", user.id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise NotFoundException("Document not found")
    return result.data[0]


async def _download_document_bytes(client: Client, document: dict) -> bytes:
    upload = get_yaml_config().upload

    async def _fetch() -> bytes:
        return await asyncio.to_thread(
            client.storage.from_(upload.storage_bucket).download,
            document["storage_path"],
        )

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
    doc = _get_owned_document(client, document_id, user)
    if doc["file_type"] != "excel":
        raise FileException("Excel analysis is only available for spreadsheet uploads")

    cache_key = excel_cache_key(document_id, doc.get("file_hash"))
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

    doc = _get_owned_document(client, document_id, user)
    if doc["file_type"] != "excel":
        raise FileException("Only spreadsheet uploads can be analyzed")
    if doc["status"] == "processing":
        raise FileException("Spreadsheet is already being analyzed", status_code=409)

    cache_key = excel_cache_key(document_id, doc.get("file_hash"))
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
        message = str(exc)
        client.table("documents").update(
            {"status": "failed", "error_message": message}
        ).eq("id", document_id).execute()
        raise FileException(f"Excel analysis failed: {message}", status_code=500) from exc


async def create_custom_excel_chart(
    client: Client,
    document_id: str,
    user: AuthUser,
    request: CustomChartRequest,
) -> dict:
    doc = _get_owned_document(client, document_id, user)
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
