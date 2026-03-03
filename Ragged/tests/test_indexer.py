import shutil
from pathlib import Path

import pytest

from ragged.config import RaggedConfig
from ragged.indexer import Indexer, _file_hash, _clean_metadata


@pytest.fixture
def tmp_storage(tmp_path):
    """Create a temp config pointing to tmp directories."""
    config = RaggedConfig()
    config.storage.chroma_path = str(tmp_path / "chroma")
    config.storage.bm25_path = str(tmp_path / "bm25")
    config.storage.hash_db_path = str(tmp_path / "hashes.db")
    return config


@pytest.fixture
def sample_docs(tmp_path):
    """Create sample documents in a temp directory."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("# Guide\n\nThis is the user guide.\n\n## Setup\n\nRun pip install.")
    (docs_dir / "notes.txt").write_text("Some plain text notes about the project.")
    (docs_dir / "image.png").write_bytes(b"\x89PNG")  # Should be skipped
    return docs_dir


class TestFileHash:
    def test_hash_is_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert _file_hash(f) == _file_hash(f)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert _file_hash(f1) != _file_hash(f2)


class TestCleanMetadata:
    def test_none_becomes_empty_string(self):
        assert _clean_metadata({"key": None}) == {"key": ""}

    def test_strings_pass_through(self):
        assert _clean_metadata({"k": "v"}) == {"k": "v"}

    def test_complex_types_become_json(self):
        result = _clean_metadata({"k": [1, 2, 3]})
        assert result["k"] == "[1, 2, 3]"


class TestIndexer:
    def test_ingest_processes_supported_files(self, tmp_storage, sample_docs):
        indexer = Indexer(tmp_storage)
        result = indexer.ingest(sample_docs)
        assert result.files_processed == 2  # md + txt, not png
        assert result.chunks_created > 0

    def test_ingest_skips_unchanged_files(self, tmp_storage, sample_docs):
        indexer = Indexer(tmp_storage)
        result1 = indexer.ingest(sample_docs)
        result2 = indexer.ingest(sample_docs)
        assert result2.files_skipped == 2
        assert result2.files_processed == 0

    def test_ingest_detects_changed_files(self, tmp_storage, sample_docs):
        indexer = Indexer(tmp_storage)
        indexer.ingest(sample_docs)
        # Modify a file
        (sample_docs / "guide.md").write_text("# Updated\n\nNew content.")
        result = indexer.ingest(sample_docs)
        assert result.files_processed == 1
        assert result.files_skipped == 1

    def test_list_collections(self, tmp_storage, sample_docs):
        indexer = Indexer(tmp_storage)
        indexer.ingest(sample_docs, collection="test_col")
        collections = indexer.list_collections()
        names = [c.name for c in collections]
        assert "test_col" in names

    def test_delete_collection(self, tmp_storage, sample_docs):
        indexer = Indexer(tmp_storage)
        indexer.ingest(sample_docs, collection="to_delete")
        indexer.delete_collection("to_delete")
        collections = indexer.list_collections()
        names = [c.name for c in collections]
        assert "to_delete" not in names

    def test_ingest_invalid_directory_raises(self, tmp_storage):
        indexer = Indexer(tmp_storage)
        with pytest.raises(ValueError, match="Not a directory"):
            indexer.ingest("/nonexistent/path")
