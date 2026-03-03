from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf

from ragged.models import Document


class PdfParser:
    """Parse PDF files, returning one Document per page."""

    def parse(self, file_path: Path) -> list[Document]:
        file_path = Path(file_path)
        documents = []

        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    documents.append(
                        Document(
                            text=text,
                            metadata={
                                "source": str(file_path),
                                "page": page_num,
                                "section": None,
                            },
                        )
                    )

        return documents
