from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ragged.models import Document


class BaseParser(Protocol):
    """Protocol that all document parsers must implement."""

    def parse(self, file_path: Path) -> list[Document]:
        """Parse a file and return a list of Documents.

        Each Document contains the text content and metadata dict with keys:
          - source: str (file path)
          - page: int | None (page number for multi-page docs)
          - section: str | None (section/heading title if available)
        """
        ...
