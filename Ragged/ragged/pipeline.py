from __future__ import annotations

from pathlib import Path

from ragged.config import RaggedConfig
from ragged.generator import Generator
from ragged.indexer import Indexer
from ragged.models import Answer
from ragged.reranker import Reranker
from ragged.retriever.bm25 import BM25Retriever
from ragged.retriever.hybrid import HybridRetriever
from ragged.retriever.vector import VectorRetriever


class RAGPipeline:
    """Full RAG pipeline: retrieve -> rerank -> generate with citations."""

    def __init__(self, config: RaggedConfig):
        self.config = config
        self._generator = Generator(
            ollama_url=config.ollama.url,
            model=config.ollama.model,
        )
        self._reranker = Reranker(model_name=config.reranker.model)
        self._indexer = Indexer(config)

    def query(self, question: str, collection: str = "default") -> Answer:
        """Run the full RAG pipeline for a question."""
        # Build retrievers
        bm25 = BM25Retriever(
            bm25_path=Path(self.config.storage.bm25_path),
            collection=collection,
        )
        vector = VectorRetriever(
            chroma_path=self.config.storage.chroma_path,
            collection=collection,
            embedding_model=self.config.embeddings.model,
        )
        hybrid = HybridRetriever(
            bm25=bm25,
            vector=vector,
            rrf_k=self.config.retrieval.rrf_k,
        )

        # Step 1: Hybrid retrieval
        candidates = hybrid.search(question, top_k=self.config.retrieval.top_k)

        if not candidates:
            return Answer(
                text="I don't have enough information to answer this based on the available documentation.",
                sources=[],
                query=question,
            )

        # Step 2: Cross-encoder reranking
        reranked = self._reranker.rerank(
            question, candidates, top_n=self.config.reranker.top_n
        )

        # Step 3: Generate answer with citations
        return self._generator.generate(question, reranked)

    @property
    def indexer(self) -> Indexer:
        return self._indexer
