from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from ragged.models import Document


class HtmlParser:
    """Parse HTML files, extracting text content."""

    def parse(self, file_path: Path) -> list[Document]:
        file_path = Path(file_path)
        raw = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Try to extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        metadata = {
            "source": str(file_path),
            "page": None,
            "section": title,
        }

        return [Document(text=text, metadata=metadata)] if text.strip() else []
