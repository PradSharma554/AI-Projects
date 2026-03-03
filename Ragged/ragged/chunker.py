from __future__ import annotations

import hashlib
import re

from ragged.models import Chunk, Document


def _split_by_headers(text: str) -> list[tuple[str | None, str]]:
    """Split text by markdown headers. Returns list of (heading, content) tuples."""
    pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str | None, str]] = []
    last_end = 0
    current_heading = None

    for match in pattern.finditer(text):
        # Content before this header
        content = text[last_end : match.start()].strip()
        if content:
            sections.append((current_heading, content))
        current_heading = match.group(2).strip()
        last_end = match.end()

    # Remaining content after last header
    remaining = text[last_end:].strip()
    if remaining:
        sections.append((current_heading, remaining))

    # If no headers found, return the whole text
    if not sections and text.strip():
        sections.append((None, text.strip()))

    return sections


def _split_preserving_code_blocks(text: str) -> list[str]:
    """Split text into segments, keeping fenced code blocks intact."""
    segments: list[str] = []
    code_fence = re.compile(r"^```", re.MULTILINE)
    matches = list(code_fence.finditer(text))

    if len(matches) < 2:
        # No complete code blocks, split on paragraphs
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    last_end = 0
    i = 0
    while i < len(matches) - 1:
        # Text before the code block
        before = text[last_end : matches[i].start()].strip()
        if before:
            segments.extend(p.strip() for p in before.split("\n\n") if p.strip())

        # The code block itself (from opening ``` to closing ```)
        code_end = text.find("\n", matches[i + 1].start())
        if code_end == -1:
            code_end = len(text)
        code_block = text[matches[i].start() : code_end].strip()
        if code_block:
            segments.append(code_block)

        last_end = code_end
        i += 2

    # Remaining text after last code block
    remaining = text[last_end:].strip()
    if remaining:
        segments.extend(p.strip() for p in remaining.split("\n\n") if p.strip())

    return segments


def _make_chunk_id(source: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{source}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split documents into chunks respecting headers, paragraphs, and code blocks.

    Uses character count as a proxy for tokens (~4 chars per token).
    """
    char_limit = chunk_size * 4  # rough chars-per-token estimate
    overlap_chars = chunk_overlap * 4
    all_chunks: list[Chunk] = []
    global_index = 0

    for doc in documents:
        sections = _split_by_headers(doc.text)

        for heading, section_text in sections:
            segments = _split_preserving_code_blocks(section_text)
            current_chunk = ""

            for segment in segments:
                # If a single segment exceeds 2x limit, force-split it
                if len(segment) > char_limit * 2:
                    # Flush current
                    if current_chunk.strip():
                        all_chunks.append(
                            _build_chunk(doc, current_chunk.strip(), heading, global_index)
                        )
                        global_index += 1
                        current_chunk = ""

                    # Force-split the long segment
                    for sub in _force_split(segment, char_limit, overlap_chars):
                        all_chunks.append(_build_chunk(doc, sub, heading, global_index))
                        global_index += 1
                    continue

                # If adding this segment would exceed the limit, flush
                if current_chunk and len(current_chunk) + len(segment) + 2 > char_limit:
                    all_chunks.append(
                        _build_chunk(doc, current_chunk.strip(), heading, global_index)
                    )
                    global_index += 1
                    # Keep overlap from the end of previous chunk
                    current_chunk = current_chunk[-overlap_chars:] if overlap_chars else ""

                if current_chunk:
                    current_chunk += "\n\n" + segment
                else:
                    current_chunk = segment

            # Flush remaining
            if current_chunk.strip():
                all_chunks.append(_build_chunk(doc, current_chunk.strip(), heading, global_index))
                global_index += 1

    return all_chunks


def _build_chunk(doc: Document, text: str, heading: str | None, index: int) -> Chunk:
    metadata = {**doc.metadata, "chunk_index": index}
    if heading:
        metadata["section"] = heading
    return Chunk(
        id=_make_chunk_id(doc.metadata.get("source", ""), index),
        text=text,
        metadata=metadata,
    )


def _force_split(text: str, char_limit: int, overlap_chars: int) -> list[str]:
    """Split long text into fixed-size pieces with overlap."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + char_limit
        piece = text[start:end]
        if piece.strip():
            pieces.append(piece.strip())
        start = end - overlap_chars if overlap_chars else end
    return pieces
