"""Tests for document storage encryption."""

import pytest
from cryptography.fernet import Fernet

from app.core import storage_crypto
from app.core.config import settings


@pytest.fixture(autouse=True)
def reset_fernet_cache():
    storage_crypto._fernet = None
    yield
    storage_crypto._fernet = None


def test_encrypt_decrypt_roundtrip(monkeypatch):
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setattr(settings, "document_storage_encryption_key", key)

    original = b"%PDF-1.4 sample content"
    encrypted = storage_crypto.encrypt_bytes(original)
    assert encrypted != original
    assert storage_crypto.decrypt_bytes(encrypted) == original


def test_storage_encryption_disabled_when_key_missing(monkeypatch):
    monkeypatch.setattr(settings, "document_storage_encryption_key", "")
    assert storage_crypto.storage_encryption_enabled() is False


def test_generate_storage_encryption_key():
    key = storage_crypto.generate_storage_encryption_key()
    assert Fernet(key.encode("ascii"))
