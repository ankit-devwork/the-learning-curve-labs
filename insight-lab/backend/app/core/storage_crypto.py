"""Application-level encryption for document blobs at rest in Supabase Storage."""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken
from pycorekit.exceptions.file import FileException

from app.core.config import settings

_fernet: Fernet | None = None


def storage_encryption_enabled() -> bool:
    return bool(settings.document_storage_encryption_key.strip())


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is not None:
        return _fernet

    raw_key = settings.document_storage_encryption_key.strip()
    if not raw_key:
        raise FileException(
            "Document storage encryption is not configured",
            status_code=500,
        )

    try:
        key_bytes = raw_key.encode("ascii")
        Fernet(key_bytes)
    except (ValueError, TypeError) as exc:
        raise FileException(
            "DOCUMENT_STORAGE_ENCRYPTION_KEY must be a valid Fernet key",
            status_code=500,
        ) from exc

    _fernet = Fernet(key_bytes)
    return _fernet


def generate_storage_encryption_key() -> str:
    """Generate a Fernet key suitable for DOCUMENT_STORAGE_ENCRYPTION_KEY."""
    return Fernet.generate_key().decode("ascii")


def encrypt_bytes(content: bytes) -> bytes:
    return _get_fernet().encrypt(content)


def decrypt_bytes(content: bytes) -> bytes:
    try:
        return _get_fernet().decrypt(content)
    except InvalidToken as exc:
        raise FileException("Stored document could not be decrypted", status_code=500) from exc
