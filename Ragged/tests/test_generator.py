from unittest.mock import patch, MagicMock

from ragged.generator import Generator
from ragged.models import Chunk, ScoredChunk


class TestGenerator:
    def test_generate_with_no_chunks_returns_no_info(self):
        gen = Generator()
        answer = gen.generate("what is X?", [])
        assert "don't have enough information" in answer.text
        assert answer.sources == []

    @patch("ragged.generator.httpx.post")
    def test_generate_calls_ollama_and_parses_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "The answer is X [1]. Also Y [2].\n\nSources:\n[1] doc.md\n[2] other.md"}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        gen = Generator(ollama_url="http://localhost:11434", model="test")
        chunks = [
            ScoredChunk(
                chunk=Chunk(id="1", text="chunk text 1", metadata={"source": "doc.md", "section": "Intro"}),
                score=0.9,
            ),
            ScoredChunk(
                chunk=Chunk(id="2", text="chunk text 2", metadata={"source": "other.md"}),
                score=0.8,
            ),
        ]
        answer = gen.generate("what is X?", chunks)

        assert "[1]" in answer.text
        assert len(answer.sources) == 2
        assert answer.sources[0].source == "doc.md"

    @patch("ragged.generator.httpx.post")
    def test_generate_handles_no_citations_in_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "I found no relevant information."}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        gen = Generator()
        chunks = [
            ScoredChunk(chunk=Chunk(id="1", text="text", metadata={"source": "x.md"}), score=0.5),
        ]
        answer = gen.generate("question?", chunks)
        assert answer.sources == []
