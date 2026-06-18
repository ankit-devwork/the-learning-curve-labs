"""User-safe error messages — never expose internal exception details to clients."""

GENERIC_PROCESSING_ERROR = "Processing failed. Please try again."
GENERIC_EXCEL_ERROR = "Spreadsheet analysis failed. Please try again."
GENERIC_UPLOAD_ERROR = "Upload failed. Please try again."


def user_facing_error_message(exc: Exception, *, fallback: str = GENERIC_PROCESSING_ERROR) -> str:
    from pycorekit.exceptions.base import AppException

    if isinstance(exc, AppException):
        return exc.message
    return fallback


def sanitize_stored_error(message: str | None) -> str | None:
    if not message:
        return None
    lowered = message.lower()
    internal_markers = (
        "traceback",
        "exception",
        "supabase",
        "postgres",
        "postgrest",
        "httpx",
        "connection",
        "embedding",
        "storage_path",
        "sqlalchemy",
    )
    if any(marker in lowered for marker in internal_markers):
        return GENERIC_PROCESSING_ERROR
    return message
