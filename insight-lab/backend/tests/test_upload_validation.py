"""Upload validation tests."""

import pytest
from pycorekit.exceptions.file import FileException

from app.services.upload_validation import sanitize_filename, validate_upload


def test_validate_pdf_upload():
    content = b"%PDF-1.4 test content"
    result = validate_upload(
        filename="report.pdf",
        content=content,
        mime_type="application/pdf",
    )
    assert result.file_type == "document"
    assert result.extension == ".pdf"


def test_rejects_invalid_extension():
    with pytest.raises(FileException, match="Unsupported file type"):
        validate_upload(filename="virus.exe", content=b"data", mime_type=None)


def test_rejects_empty_file():
    with pytest.raises(FileException, match="empty"):
        validate_upload(filename="report.pdf", content=b"", mime_type="application/pdf")


def test_rejects_file_over_limit(monkeypatch):
    from app.core import yaml_config

    config = yaml_config.get_yaml_config()
    monkeypatch.setattr(config.upload, "max_bytes", 10)

    with pytest.raises(FileException, match="too large"):
        validate_upload(filename="report.pdf", content=b"%PDF-1234567890", mime_type="application/pdf")


def test_rejects_bad_pdf_signature():
    with pytest.raises(FileException, match="does not match expected format"):
        validate_upload(filename="report.pdf", content=b"NOTPDF", mime_type="application/pdf")


def test_sanitize_filename():
    assert sanitize_filename("My File (1).pdf", max_length=255) == "My File _1_.pdf"
