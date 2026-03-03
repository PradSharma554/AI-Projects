from unittest.mock import MagicMock, patch

from ragged.models import Chunk, ScoredChunk
from ragged.reranker import Reranker


class TestReranker:
    @patch("ragged.reranker.CrossEncoder")
    def test_rerank_returns_top_n(self, mock_ce_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1, 0.5]
        mock_ce_cls.return_value = mock_model

        reranker = Reranker(model_name="test-model")
        chunks = [
            ScoredChunk(chunk=Chunk(id="a", text="text a", metadata={}), score=0.0),
            ScoredChunk(chunk=Chunk(id="b", text="text b", metadata={}), score=0.0),
            ScoredChunk(chunk=Chunk(id="c", text="text c", metadata={}), score=0.0),
        ]
        result = reranker.rerank("query", chunks, top_n=2)
        assert len(result) == 2
        assert result[0].chunk.id == "a"  # highest score 0.9
        assert result[1].chunk.id == "c"  # second highest 0.5

    @patch("ragged.reranker.CrossEncoder")
    def test_rerank_empty_chunks(self, mock_ce_cls):
        reranker = Reranker(model_name="test-model")
        result = reranker.rerank("query", [], top_n=5)
        assert result == []

    @patch("ragged.reranker.CrossEncoder")
    def test_rerank_scores_assigned(self, mock_ce_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.7]
        mock_ce_cls.return_value = mock_model

        reranker = Reranker(model_name="test-model")
        chunks = [ScoredChunk(chunk=Chunk(id="x", text="text", metadata={}), score=0.0)]
        result = reranker.rerank("query", chunks, top_n=1)
        assert result[0].score == 0.7
