from __future__ import annotations

from sentence_transformers import CrossEncoder

from ragged.models import ScoredChunk


class Reranker:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_n: int = 5
    ) -> list[ScoredChunk]:
        """Score each (query, chunk) pair and return top_n by relevance."""
        if not chunks:
            return []

        pairs = [[query, sc.chunk.text] for sc in chunks]
        scores = self._model.predict(pairs)

        scored = []
        for sc, score in zip(chunks, scores):
            scored.append(ScoredChunk(chunk=sc.chunk, score=float(score)))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]
