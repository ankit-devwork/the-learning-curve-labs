import io
from pathlib import Path

from pycorekit.exceptions.file import FileException


def extract_text_from_bytes(content: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(content)
    if ext == ".txt":
        return _extract_txt(content)
    if ext == ".docx":
        return _extract_docx(content)
    if ext == ".doc":
        raise FileException("Legacy .doc files are not supported yet; please upload .docx or .pdf")

    raise FileException(f"Text extraction is not supported for '{ext}'")


def _extract_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise FileException("PDF parser not installed (pypdf)", status_code=500) from exc

    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise FileException("Could not extract text from PDF (it may be scanned/image-only)")
    return text


def _extract_txt(content: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            text = content.decode(encoding).strip()
            if text:
                return text
        except UnicodeDecodeError:
            continue
    raise FileException("Could not decode text file")


def _extract_docx(content: bytes) -> str:
    try:
        import docx
    except ImportError as exc:
        raise FileException("DOCX parser not installed (python-docx)", status_code=500) from exc

    document = docx.Document(io.BytesIO(content))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs).strip()
    if not text:
        raise FileException("Could not extract text from DOCX")
    return text
