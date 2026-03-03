from __future__ import annotations

from pathlib import Path

from ragged.models import Document


class TextParser:
    """Parse plain text files."""

    def parse(self, file_path: Path) -> list[Document]:
        file_path = Path(file_path)
        text = file_path.read_text(encoding="utf-8")

        metadata = {
            "source": str(file_path),
            "page": None,
            "section": None,
        }

        return [Document(text=text, metadata=metadata)] if text.strip() else []
