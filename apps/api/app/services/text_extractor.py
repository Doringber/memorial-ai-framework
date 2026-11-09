"""Extract plain text from various file formats."""

from pathlib import Path

from loguru import logger


def extract_text(file_path: str, source_type: str = "") -> str:
    """Extract text from a file. Supports txt, md, pdf, docx, and WhatsApp exports."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if not source_type:
        source_type = suffix.lstrip(".")

    if source_type in ("txt", "md", "text", "whatsapp"):
        return _extract_plain(path)
    elif source_type == "pdf":
        return _extract_pdf(path)
    elif source_type in ("docx", "doc"):
        return _extract_docx(path)
    else:
        logger.warning("Unknown source type '{}', trying plain text", source_type)
        return _extract_plain(path)


def _extract_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
