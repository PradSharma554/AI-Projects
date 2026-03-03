from ragged.models import Chunk, ScoredChunk
from ragged.retriever.hybrid import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_merges_two_lists(self):
        chunk_a = Chunk(id="a", text="chunk a", metadata={})
        chunk_b = Chunk(id="b", text="chunk b", metadata={})
        chunk_c = Chunk(id="c", text="chunk c", metadata={})

        list1 = [ScoredChunk(chunk=chunk_a, score=1.0), ScoredChunk(chunk=chunk_b, score=0.5)]
        list2 = [ScoredChunk(chunk=chunk_b, score=1.0), ScoredChunk(chunk=chunk_c, score=0.5)]

        results = reciprocal_rank_fusion([list1, list2], rrf_k=60, top_k=10)

        ids = [r.chunk.id for r in results]
        # chunk_b appears in both lists, should be ranked highest
        assert ids[0] == "b"
        assert len(results) == 3

    def test_respects_top_k(self):
        chunks = [
            ScoredChunk(chunk=Chunk(id=f"c{i}", text=f"text {i}", metadata={}), score=1.0)
            for i in range(10)
        ]

        results = reciprocal_rank_fusion([chunks], rrf_k=60, top_k=3)
        assert len(results) == 3

    def test_empty_lists_return_empty(self):
        results = reciprocal_rank_fusion([[], []], rrf_k=60, top_k=10)
        assert results == []

    def test_single_list(self):
        chunk = Chunk(id="x", text="only one", metadata={})
        results = reciprocal_rank_fusion(
            [[ScoredChunk(chunk=chunk, score=1.0)]], rrf_k=60, top_k=10
        )
        assert len(results) == 1
        assert results[0].chunk.id == "x"

    def test_duplicate_across_lists_accumulates_score(self):
        chunk = Chunk(id="dup", text="duplicate", metadata={})
        list1 = [ScoredChunk(chunk=chunk, score=1.0)]
        list2 = [ScoredChunk(chunk=chunk, score=1.0)]

        results_single = reciprocal_rank_fusion([list1], rrf_k=60, top_k=10)
        results_both = reciprocal_rank_fusion([list1, list2], rrf_k=60, top_k=10)

        # Score should be higher when appearing in both lists
        assert results_both[0].score > results_single[0].score
