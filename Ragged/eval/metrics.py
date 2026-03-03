from __future__ import annotations

import re


def retrieval_recall_at_k(
    retrieved_sources: list[str], expected_sources: list[str]
) -> float:
    """Compute recall@K: fraction of expected sources found in retrieved results."""
    if not expected_sources:
        return 1.0
    retrieved_set = set(retrieved_sources)
    hits = sum(1 for s in expected_sources if any(s in r for r in retrieved_set))
    return hits / len(expected_sources)


def citation_accuracy(answer_text: str, sources: list[dict], expected_sources: list[str]) -> float:
    """Compute what fraction of citations [N] map to expected source documents."""
    cited_indices = set(int(m) for m in re.findall(r"\[(\d+)\]", answer_text))
    if not cited_indices:
        return 0.0

    correct = 0
    for idx in cited_indices:
        # Find the source for this citation index
        matching = [s for s in sources if s.get("index") == idx]
        if matching:
            src_path = matching[0].get("source", "")
            if any(exp in src_path for exp in expected_sources):
                correct += 1

    return correct / len(cited_indices)


def parse_faithfulness_score(llm_response: str) -> float:
    """Extract a 0-1 score from an LLM-as-judge faithfulness response."""
    # Look for patterns like "Score: 0.8" or "0.85" or "85%"
    score_match = re.search(r"(?:score[:\s]*)?(\d+(?:\.\d+)?)\s*(?:%|/\s*(?:10|100))?", llm_response.lower())
    if score_match:
        val = float(score_match.group(1))
        if val > 1:
            val = val / 100.0
        return min(max(val, 0.0), 1.0)
    return 0.0


FAITHFULNESS_JUDGE_PROMPT = """You are evaluating the faithfulness of an AI-generated answer.

Given the context chunks and the answer, score how well the answer is supported by the context.

Rules:
- Score 1.0 if every claim in the answer is directly supported by the context
- Score 0.5 if some claims are supported but others are not
- Score 0.0 if the answer contains claims not found in the context

Context:
{context}

Answer:
{answer}

Respond with ONLY a score between 0.0 and 1.0, like: Score: 0.85"""
