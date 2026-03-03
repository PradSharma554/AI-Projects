from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from ragged.models import Chunk, ScoredChunk


class BM25Retriever:
    """Keyword-based retrieval using BM25."""

    def __init__(self, bm25_path: Path, collection: str = "default"):
        self._corpus_file = Path(bm25_path) / f"{collection}.pkl"
        self._corpus: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        if not self._corpus_file.exists():
            return
        with open(self._corpus_file, "rb") as f:
            self._corpus = pickle.load(f)
        if self._corpus:
            tokenized = [item["text"].lower().split() for item in self._corpus]
            self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        if not self._bm25 or not self._corpus:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in scored_indices:
            if score <= 0:
                continue
            item = self._corpus[idx]
            chunk = Chunk(id=item["id"], text=item["text"], metadata=item["metadata"])
            results.append(ScoredChunk(chunk=chunk, score=float(score)))

        return results
