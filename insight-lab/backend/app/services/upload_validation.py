import re
from dataclasses import dataclass
from pathlib import Path

from pycorekit.exceptions.file import FileException

from app.core.yaml_config import UploadSection, get_yaml_config

GENERIC_MIME_TYPES = frozenset(
    {
        "application/octet-stream",
        "binary/octet-stream",
        "application/x-msdownload",
    }
)


@dataclass(frozen=True)
class ValidatedUpload:
    safe_filename: str
    file_type: str
    extension: str


def _upload_config() -> UploadSection:
    return get_yaml_config().upload


def sanitize_filename(filename: str, *, max_length: int) -> str:
    base = Path(filename).name.strip()
    if not base:
        raise FileException("Filename is required")
    cleaned = re.sub(r"[^\w.\- ]+", "_", base)
    return cleaned[:max_length]


def _validate_extension(safe_name: str, upload: UploadSection) -> tuple[str, str]:
    extension = Path(safe_name).suffix.lower()
    if not extension:
        raise FileException("File must have an extension")

    file_type = upload.file_type_for_extension(extension)
    if file_type is None:
        allowed = ", ".join(sorted(upload.all_extensions()))
        raise FileException(f"Unsupported file type. Allowed extensions: {allowed}")

    return extension, file_type


def _validate_size(content: bytes, upload: UploadSection) -> None:
    if not content:
        raise FileException("Uploaded file is empty")

    if len(content) > upload.max_bytes:
        max_mb = upload.max_bytes / (1024 * 1024)
        raise FileException(f"File too large. Maximum size is {max_mb:g} MB")


def _validate_mime_type(
    mime_type: str | None,
    file_type: str,
    upload: UploadSection,
) -> None:
    if not mime_type or mime_type.lower() in GENERIC_MIME_TYPES:
        return

    type_config = upload.type_config(file_type)
    if not type_config.mime_types:
        return

    normalized = mime_type.lower().split(";")[0].strip()
    allowed = {mime.lower() for mime in type_config.mime_types}
    if normalized not in allowed:
        raise FileException(
            f"MIME type '{mime_type}' is not allowed for {file_type} uploads"
        )


def _validate_signature(content: bytes, extension: str, file_type: str, upload: UploadSection) -> None:
    type_config = upload.type_config(file_type)
    signature_hex = type_config.signatures.get(extension, "")
    if not signature_hex:
        return

    expected = signature_hex.lower().replace(" ", "")
    actual = content[: len(expected) // 2].hex()
    if not actual.startswith(expected):
        raise FileException(
            f"File content does not match expected format for '{extension}' uploads"
        )


def validate_upload(
    *,
    filename: str,
    content: bytes,
    mime_type: str | None,
) -> ValidatedUpload:
    upload = _upload_config()
    safe_name = sanitize_filename(filename, max_length=upload.filename_max_length)
    extension, file_type = _validate_extension(safe_name, upload)
    _validate_size(content, upload)
    _validate_mime_type(mime_type, file_type, upload)
    _validate_signature(content, extension, file_type, upload)

    return ValidatedUpload(
        safe_filename=safe_name,
        file_type=file_type,
        extension=extension,
    )
