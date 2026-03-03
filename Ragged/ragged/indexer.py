from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ragged.chunker import chunk_documents
from ragged.config import RaggedConfig
from ragged.models import CollectionInfo, IngestResult
from ragged.parsers import SUPPORTED_EXTENSIONS, get_parser


class Indexer:
    """Orchestrates document ingestion into ChromaDB (vectors) and BM25 (keywords)."""

    def __init__(self, config: RaggedConfig):
        self.config = config
        self._chroma_client = chromadb.PersistentClient(path=config.storage.chroma_path)
        self._embedder = SentenceTransformer(config.embeddings.model)
        self._hash_db = _HashDB(config.storage.hash_db_path)
        self._bm25_path = Path(config.storage.bm25_path)
        self._bm25_path.mkdir(parents=True, exist_ok=True)

    def ingest(self, directory: str | Path, collection: str = "default") -> IngestResult:
        """Parse all supported docs in directory, chunk, embed, store."""
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        chroma_col = self._chroma_client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

        files_processed = 0
        files_skipped = 0
        all_chunks = []

        for file_path in sorted(directory.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            # Check if file changed
            file_hash = _file_hash(file_path)
            if self._hash_db.is_unchanged(str(file_path), file_hash, collection):
                files_skipped += 1
                continue

            parser = get_parser(file_path)
            if parser is None:
                continue

            docs = parser.parse(file_path)
            chunks = chunk_documents(
                docs,
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap,
            )
            all_chunks.extend(chunks)
            files_processed += 1
            self._hash_db.update(str(file_path), file_hash, collection)

        if not all_chunks:
            return IngestResult(
                files_processed=files_processed,
                chunks_created=0,
                files_skipped=files_skipped,
            )

        # Embed and store in ChromaDB
        texts = [c.text for c in all_chunks]
        ids = [c.id for c in all_chunks]
        metadatas = [c.metadata for c in all_chunks]

        # ChromaDB metadata values must be str, int, float, or bool
        clean_metadatas = [_clean_metadata(m) for m in metadatas]

        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()

        # Upsert in batches (ChromaDB has batch limits)
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            chroma_col.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=clean_metadatas[i:end],
            )

        # Save BM25 corpus
        self._save_bm25_corpus(collection, all_chunks)

        return IngestResult(
            files_processed=files_processed,
            chunks_created=len(all_chunks),
            files_skipped=files_skipped,
        )

    def delete_collection(self, collection: str) -> None:
        """Delete a collection from ChromaDB and BM25."""
        self._chroma_client.delete_collection(name=collection)
        bm25_file = self._bm25_path / f"{collection}.pkl"
        if bm25_file.exists():
            bm25_file.unlink()
        self._hash_db.clear_collection(collection)

    def list_collections(self) -> list[CollectionInfo]:
        """List all indexed collections."""
        collections = self._chroma_client.list_collections()
        result = []
        for col_name in collections:
            # ChromaDB may return Collection objects or strings depending on version
            name = col_name if isinstance(col_name, str) else col_name.name
            col = self._chroma_client.get_collection(name=name)
            result.append(
                CollectionInfo(
                    name=name,
                    chunk_count=col.count(),
                )
            )
        return result

    def _save_bm25_corpus(self, collection: str, chunks) -> None:
        """Save chunk texts + metadata for BM25 index building."""
        corpus_data = [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks]
        bm25_file = self._bm25_path / f"{collection}.pkl"

        # Merge with existing if present
        existing = []
        if bm25_file.exists():
            with open(bm25_file, "rb") as f:
                existing = pickle.load(f)

        existing_ids = {item["id"] for item in existing}
        for item in corpus_data:
            if item["id"] not in existing_ids:
                existing.append(item)

        with open(bm25_file, "wb") as f:
            pickle.dump(existing, f)


def _file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _clean_metadata(meta: dict) -> dict:
    """Ensure all metadata values are ChromaDB-compatible types."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = json.dumps(v)
    return clean


class _HashDB:
    """SQLite-backed file hash tracker for incremental ingestion."""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT,
                collection TEXT,
                hash TEXT,
                PRIMARY KEY (file_path, collection)
            )"""
        )
        self._conn.commit()

    def is_unchanged(self, file_path: str, file_hash: str, collection: str) -> bool:
        row = self._conn.execute(
            "SELECT hash FROM file_hashes WHERE file_path = ? AND collection = ?",
            (file_path, collection),
        ).fetchone()
        return row is not None and row[0] == file_hash

    def update(self, file_path: str, file_hash: str, collection: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO file_hashes (file_path, collection, hash) VALUES (?, ?, ?)",
            (file_path, collection, file_hash),
        )
        self._conn.commit()

    def clear_collection(self, collection: str) -> None:
        self._conn.execute("DELETE FROM file_hashes WHERE collection = ?", (collection,))
        self._conn.commit()
