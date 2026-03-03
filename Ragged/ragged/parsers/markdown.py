from __future__ import annotations

from pathlib import Path

import frontmatter

from ragged.models import Document


class MarkdownParser:
    """Parse Markdown files with optional YAML frontmatter."""

    def parse(self, file_path: Path) -> list[Document]:
        file_path = Path(file_path)
        raw = file_path.read_text(encoding="utf-8")
        post = frontmatter.loads(raw)

        metadata = {
            "source": str(file_path),
            "page": None,
            "section": None,
        }

        # Include frontmatter keys like title in metadata
        if post.metadata.get("title"):
            metadata["section"] = post.metadata["title"]

        return [Document(text=post.content, metadata=metadata)]
