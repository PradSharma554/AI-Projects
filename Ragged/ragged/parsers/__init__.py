from __future__ import annotations

from pathlib import Path

from ragged.parsers.base import BaseParser
from ragged.parsers.docx import DocxParser
from ragged.parsers.html import HtmlParser
from ragged.parsers.markdown import MarkdownParser
from ragged.parsers.pdf import PdfParser
from ragged.parsers.text import TextParser

PARSER_REGISTRY: dict[str, BaseParser] = {
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".pdf": PdfParser(),
    ".html": HtmlParser(),
    ".htm": HtmlParser(),
    ".docx": DocxParser(),
    ".txt": TextParser(),
    ".text": TextParser(),
    ".rst": TextParser(),
    ".log": TextParser(),
}

SUPPORTED_EXTENSIONS = set(PARSER_REGISTRY.keys())


def get_parser(file_path: str | Path) -> BaseParser | None:
    """Return the appropriate parser for a file based on its extension."""
    ext = Path(file_path).suffix.lower()
    return PARSER_REGISTRY.get(ext)
