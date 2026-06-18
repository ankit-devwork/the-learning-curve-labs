from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from pycorekit.exceptions.file import FileException
from pycorekit.tracing.decorators import with_observability

from app.core.auth import AuthUser
from app.core.cache import check_rate_limit
from app.core.deps import get_current_user
from app.core.exceptions import RateLimitException
from app.core.supabase_client import get_supabase_client
from app.core.yaml_config import get_yaml_config
from app.services.upload import get_upload_public_config, list_documents, upload_document
from app.services.upload_validation import validate_upload

router = APIRouter()


@router.get("/upload/config")
@with_observability("upload_config")
async def upload_config(request: Request):
    correlation_id = getattr(request.state, "correlation_id", None)
    return {
        **get_upload_public_config(),
        "correlation_id": correlation_id,
    }


@router.post("/upload")
@with_observability("upload_document")
async def upload_file(
    request: Request,
    user: AuthUser = Depends(get_current_user),
    file: UploadFile = File(...),
):
    if not file.filename:
        raise FileException("Filename is required")

    upload_cfg = get_yaml_config().upload
    allowed, retry_after = await check_rate_limit(
        key=f"upload:{user.id}",
        limit=upload_cfg.rate_limit_per_hour,
        window_seconds=3600,
    )
    if not allowed:
        raise RateLimitException(
            f"Upload rate limit reached ({upload_cfg.rate_limit_per_hour}/hour)",
            retry_after=retry_after,
        )

    content = await file.read()
    validated = validate_upload(
        filename=file.filename,
        content=content,
        mime_type=file.content_type,
    )

    document = upload_document(
        get_supabase_client(),
        user,
        validated=validated,
        content=content,
        mime_type=file.content_type,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {
        **document,
        "correlation_id": correlation_id,
    }


@router.get("/documents")
@with_observability("list_documents")
async def get_documents(
    request: Request,
    user: AuthUser = Depends(get_current_user),
    limit: int | None = Query(default=None, ge=1, le=100),
):
    default_limit = get_yaml_config().upload.documents_list_default_limit
    documents = list_documents(
        get_supabase_client(),
        user,
        limit=limit if limit is not None else default_limit,
    )
    correlation_id = getattr(request.state, "correlation_id", None)
    return {
        "documents": documents,
        "count": len(documents),
        "correlation_id": correlation_id,
    }
