"""
pdf_loader.py
Extract raw text from PDF or plain-text uploads.
"""

import fitz  # PyMuPDF
from pathlib import Path


def extract_text(file_path: str) -> str:
    """
    Extract full text from a PDF or .txt/.md file.
    Returns the raw string, or raises ValueError if nothing extracted.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(path: Path) -> str:
    text_parts = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    text = "\n".join(text_parts).strip()
    if not text:
        raise ValueError(f"No text could be extracted from {path.name}. It may be a scanned image PDF.")
    return text


def get_page_count(file_path: str) -> int:
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return 1
    with fitz.open(str(path)) as doc:
        return len(doc)