from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A parsed document (or page) before chunking."""

    text: str
    metadata: dict = Field(default_factory=dict)
    # metadata keys: source (file path), page (int|None), section (str|None)


class Chunk(BaseModel):
    """A text chunk ready for indexing."""

    id: str
    text: str
    metadata: dict = Field(default_factory=dict)
    # metadata keys: source, page, section, chunk_index


class ScoredChunk(BaseModel):
    """A chunk with a relevance score from retrieval or reranking."""

    chunk: Chunk
    score: float = 0.0


class Source(BaseModel):
    """A citation source mapping."""

    index: int
    source: str
    section: str | None = None
    page: int | None = None
    text: str = ""


class Answer(BaseModel):
    """The final answer with citations."""

    text: str
    sources: list[Source] = Field(default_factory=list)
    query: str = ""


class IngestResult(BaseModel):
    """Stats returned after ingestion."""

    files_processed: int = 0
    chunks_created: int = 0
    files_skipped: int = 0


class CollectionInfo(BaseModel):
    """Info about an indexed collection."""

    name: str
    document_count: int = 0
    chunk_count: int = 0
