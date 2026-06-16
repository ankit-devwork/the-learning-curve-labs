from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from pycorekit.exceptions.file import FileException
from pycorekit.tracing.decorators import with_observability

from app.core.auth import AuthUser
from app.core.deps import get_current_user
from app.core.supabase_client import get_supabase_client
from app.services.upload import list_documents, upload_document

router = APIRouter()


@router.post("/upload")
@with_observability("upload_document")
async def upload_file(
    request: Request,
    user: AuthUser = Depends(get_current_user),
    file: UploadFile = File(...),
):
    if not file.filename:
        raise FileException("Filename is required")

    content = await file.read()
    document = upload_document(
        get_supabase_client(),
        user,
        filename=file.filename,
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
    limit: int = Query(default=20, ge=1, le=100),
):
    documents = list_documents(get_supabase_client(), user, limit=limit)
    correlation_id = getattr(request.state, "correlation_id", None)
    return {
        "documents": documents,
        "count": len(documents),
        "correlation_id": correlation_id,
    }
