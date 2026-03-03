from __future__ import annotations

from ragged.models import ScoredChunk
from ragged.retriever.bm25 import BM25Retriever
from ragged.retriever.vector import VectorRetriever


class HybridRetriever:
    """Combines BM25 and vector retrieval using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        rrf_k: int = 60,
    ):
        self._bm25 = bm25
        self._vector = vector
        self._rrf_k = rrf_k

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        bm25_results = self._bm25.search(query, top_k=top_k)
        vector_results = self._vector.search(query, top_k=top_k)

        return reciprocal_rank_fusion(
            result_lists=[bm25_results, vector_results],
            rrf_k=self._rrf_k,
            top_k=top_k,
        )


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]],
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[ScoredChunk]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d: score(d) = Σ 1 / (k + rank_i(d))
    where rank_i is the 1-based rank in each result list.
    """
    # Map chunk_id -> (best chunk object, accumulated RRF score)
    scores: dict[str, tuple[ScoredChunk, float]] = {}

    for results in result_lists:
        for rank, scored_chunk in enumerate(results, start=1):
            chunk_id = scored_chunk.chunk.id
            rrf_score = 1.0 / (rrf_k + rank)

            if chunk_id in scores:
                existing_chunk, existing_score = scores[chunk_id]
                scores[chunk_id] = (existing_chunk, existing_score + rrf_score)
            else:
                scores[chunk_id] = (scored_chunk, rrf_score)

    # Sort by RRF score descending
    ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)[:top_k]

    return [ScoredChunk(chunk=sc.chunk, score=rrf_score) for sc, rrf_score in ranked]
