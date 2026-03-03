from ragged.chunker import chunk_documents, _split_by_headers, _split_preserving_code_blocks
from ragged.models import Document


def _make_doc(text, source="test.md"):
    return Document(text=text, metadata={"source": source, "page": None, "section": None})


class TestSplitByHeaders:
    def test_splits_on_markdown_headers(self):
        text = "# Header 1\nSome content\n\n## Header 2\nMore content"
        sections = _split_by_headers(text)
        assert len(sections) == 2
        assert sections[0][0] == "Header 1"
        assert "Some content" in sections[0][1]
        assert sections[1][0] == "Header 2"

    def test_text_before_first_header(self):
        text = "Intro text\n\n# Header\nContent"
        sections = _split_by_headers(text)
        assert len(sections) == 2
        assert sections[0][0] is None
        assert "Intro" in sections[0][1]

    def test_no_headers_returns_whole_text(self):
        text = "Just some plain text\nwith newlines"
        sections = _split_by_headers(text)
        assert len(sections) == 1
        assert sections[0][0] is None


class TestSplitPreservingCodeBlocks:
    def test_keeps_code_blocks_intact(self):
        text = "Before code\n\n```python\ndef foo():\n    pass\n```\n\nAfter code"
        segments = _split_preserving_code_blocks(text)
        # Should have: before, code block, after
        code_segments = [s for s in segments if "def foo" in s]
        assert len(code_segments) == 1
        assert "```" in code_segments[0]

    def test_no_code_blocks_splits_on_paragraphs(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        segments = _split_preserving_code_blocks(text)
        assert len(segments) == 3


class TestChunkDocuments:
    def test_basic_chunking(self):
        doc = _make_doc("# Title\n\nParagraph one.\n\nParagraph two.")
        chunks = chunk_documents([doc], chunk_size=512)
        assert len(chunks) >= 1
        assert all(c.id for c in chunks)

    def test_chunk_metadata_preserved(self):
        doc = _make_doc("Some content here.", source="docs/guide.md")
        chunks = chunk_documents([doc])
        assert chunks[0].metadata["source"] == "docs/guide.md"
        assert "chunk_index" in chunks[0].metadata

    def test_heading_becomes_section_in_metadata(self):
        doc = _make_doc("# My Section\n\nContent under section.")
        chunks = chunk_documents([doc])
        assert chunks[0].metadata["section"] == "My Section"

    def test_large_text_produces_multiple_chunks(self):
        # ~4000 chars = ~1000 tokens, should split with chunk_size=100
        long_text = "Word " * 800
        doc = _make_doc(long_text)
        chunks = chunk_documents([doc], chunk_size=100)
        assert len(chunks) > 1

    def test_multiple_documents(self):
        docs = [_make_doc("Doc one content.", source="a.md"), _make_doc("Doc two content.", source="b.md")]
        chunks = chunk_documents(docs)
        sources = {c.metadata["source"] for c in chunks}
        assert "a.md" in sources
        assert "b.md" in sources

    def test_empty_document_produces_no_chunks(self):
        doc = _make_doc("")
        chunks = chunk_documents([doc])
        assert len(chunks) == 0

    def test_chunk_ids_are_unique(self):
        doc = _make_doc("Word " * 800)
        chunks = chunk_documents([doc], chunk_size=100)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))
