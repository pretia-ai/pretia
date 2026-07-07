"""Load document corpus context for RAG-aware input generation."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
_MAX_WORDS_PER_DOC = 1000
_MAX_CORPUS_WORDS = 10000


def load_corpus_context(path: str) -> str:
    """Load corpus context from a file or directory.

    If *path* is a file, return its text content (treated as a pre-written summary).
    If *path* is a directory, scan for .txt/.md/.pdf files and build a summary
    from the first ~200 words of each document.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus path not found: {path}")

    if p.is_file():
        return p.read_text(encoding="utf-8")

    return _summarize_directory(p)


def _summarize_directory(directory: Path) -> str:
    """Scan a directory for documents and return a formatted summary."""
    files = sorted(
        f
        for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    if not files:
        return ""

    per_doc_budget = min(_MAX_WORDS_PER_DOC, _MAX_CORPUS_WORDS // max(len(files), 1))

    parts: list[str] = []
    for fp in files:
        excerpt = _extract_excerpt(fp, max_words=per_doc_budget)
        if excerpt:
            parts.append(f"Document: {fp.name}\n{excerpt}")

    return "\n\n".join(parts)


def _extract_excerpt(fp: Path, max_words: int = _MAX_WORDS_PER_DOC) -> str:
    """Extract the first *max_words* words from a file."""
    if fp.suffix.lower() == ".pdf":
        return _extract_pdf_excerpt(fp, max_words)

    try:
        text = fp.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        logger.warning("Could not read %s, skipping", fp.name)
        return ""

    return _truncate_words(text, max_words)


def _extract_pdf_excerpt(fp: Path, max_words: int = _MAX_WORDS_PER_DOC) -> str:
    """Extract text from a PDF using pdfplumber (optional dependency)."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning(
            "pdfplumber not installed. Skipping %s. Install with: pip install pdfplumber",
            fp.name,
        )
        return ""

    try:
        with pdfplumber.open(fp) as pdf:
            text_parts: list[str] = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                if sum(len(t.split()) for t in text_parts) >= max_words:
                    break
            return _truncate_words(" ".join(text_parts), max_words)
    except Exception:
        logger.warning("Failed to extract text from %s, skipping", fp.name)
        return ""


def _truncate_words(text: str, max_words: int) -> str:
    """Truncate text to approximately *max_words* words."""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."
