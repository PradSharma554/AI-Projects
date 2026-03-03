from __future__ import annotations

import re

import httpx

from ragged.models import Answer, ScoredChunk, Source
from ragged.prompts import SYSTEM_PROMPT, build_user_prompt


class Generator:
    """Generate answers with citations using Ollama."""

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self._url = ollama_url.rstrip("/")
        self._model = model

    def generate(self, query: str, context_chunks: list[ScoredChunk]) -> Answer:
        """Build prompt, call Ollama, parse response into Answer with citations."""
        if not context_chunks:
            return Answer(
                text="I don't have enough information to answer this based on the available documentation.",
                sources=[],
                query=query,
            )

        # Build context for prompt
        chunk_dicts = []
        source_map: list[Source] = []
        for i, sc in enumerate(context_chunks, start=1):
            chunk_dicts.append({
                "text": sc.chunk.text,
                "source": sc.chunk.metadata.get("source", "unknown"),
                "section": sc.chunk.metadata.get("section", ""),
            })
            source_map.append(
                Source(
                    index=i,
                    source=sc.chunk.metadata.get("source", "unknown"),
                    section=sc.chunk.metadata.get("section"),
                    page=sc.chunk.metadata.get("page"),
                    text=sc.chunk.text[:200],
                )
            )

        user_prompt = build_user_prompt(query, chunk_dicts)

        # Call Ollama API
        response_text = self._call_ollama(user_prompt)

        # Extract which citations were actually used
        used_indices = set(int(m) for m in re.findall(r"\[(\d+)\]", response_text))
        used_sources = [s for s in source_map if s.index in used_indices]

        return Answer(text=response_text, sources=used_sources, query=query)

    def _call_ollama(self, user_prompt: str) -> str:
        """Make a synchronous request to Ollama's chat API."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

        response = httpx.post(
            f"{self._url}/api/chat",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")
