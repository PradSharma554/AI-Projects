from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from ragged.models import Chunk, ScoredChunk


class VectorRetriever:
    """Semantic similarity retrieval using ChromaDB + sentence-transformers."""

    def __init__(
        self,
        chroma_path: str,
        collection: str = "default",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._client.get_collection(name=collection)
        self._embedder = SentenceTransformer(embedding_model)

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        query_embedding = self._embedder.encode(query).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        scored_chunks = []
        if not results["ids"] or not results["ids"][0]:
            return scored_chunks

        for i, chunk_id in enumerate(results["ids"][0]):
            text = results["documents"][0][i] if results["documents"] else ""
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 1.0

            # ChromaDB returns cosine distance; convert to similarity score
            score = 1.0 - distance

            chunk = Chunk(id=chunk_id, text=text, metadata=metadata)
            scored_chunks.append(ScoredChunk(chunk=chunk, score=score))

        return scored_chunks
