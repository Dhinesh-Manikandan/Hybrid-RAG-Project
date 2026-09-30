"""
loader.py -- Generic document loader.

Supports: .txt, .pdf, .docx, .md
Adding a new file type requires adding one handler here only.
"""

from pathlib import Path
from typing import Optional


def load_document(path: Path) -> str:
    """
    Load raw text from a document file.

    Supported formats: .txt, .md, .pdf, .docx

    Args:
        path: Absolute path to the document file.

    Returns:
        Raw text string.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()

    if ext in {".txt", ".md"}:
        return _load_text(path)
    elif ext == ".pdf":
        return _load_pdf(path)
    elif ext == ".docx":
        return _load_docx(path)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: .txt, .md, .pdf, .docx"
        )


def _load_text(path: Path) -> str:
    """Load plain text or markdown file."""
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    """Load PDF using PyMuPDF (fitz) or pdfplumber as fallback."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = [
                page.extract_text() or ""
                for page in pdf.pages
            ]
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError(
            "PDF loading requires PyMuPDF or pdfplumber. "
            "Install with: pip install PyMuPDF  or  pip install pdfplumber"
        )


def _load_docx(path: Path) -> str:
    """Load DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(str(path))
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)
    except ImportError:
        raise ImportError(
            "DOCX loading requires python-docx. "
            "Install with: pip install python-docx"
        )
