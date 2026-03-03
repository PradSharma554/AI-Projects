"""Evaluation runner for the RAG pipeline.

Usage:
    python eval/runner.py --config config.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import yaml
from tabulate import tabulate

from eval.metrics import (
    FAITHFULNESS_JUDGE_PROMPT,
    citation_accuracy,
    parse_faithfulness_score,
    retrieval_recall_at_k,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragged.config import load_config
from ragged.indexer import Indexer
from ragged.retriever.bm25 import BM25Retriever
from ragged.retriever.hybrid import HybridRetriever
from ragged.retriever.vector import VectorRetriever


def load_golden_qa(path: str) -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("questions", [])


def evaluate_retrieval(config, question: str, expected_sources: list[str], collection: str = "default") -> tuple[list, float]:
    """Run retrieval and compute recall@K."""
    bm25 = BM25Retriever(Path(config.storage.bm25_path), collection=collection)
    vector = VectorRetriever(
        chroma_path=config.storage.chroma_path,
        collection=collection,
        embedding_model=config.embeddings.model,
    )
    hybrid = HybridRetriever(bm25=bm25, vector=vector, rrf_k=config.retrieval.rrf_k)
    results = hybrid.search(question, top_k=config.retrieval.top_k)

    retrieved_sources = [r.chunk.metadata.get("source", "") for r in results]
    recall = retrieval_recall_at_k(retrieved_sources, expected_sources)
    return results, recall


def evaluate_faithfulness(config, answer_text: str, context_text: str) -> float:
    """Use Ollama as judge to evaluate faithfulness."""
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=context_text, answer=answer_text)
    try:
        resp = httpx.post(
            f"{config.ollama.url}/api/generate",
            json={"model": config.ollama.model, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        resp.raise_for_status()
        llm_output = resp.json().get("response", "")
        return parse_faithfulness_score(llm_output)
    except Exception as e:
        print(f"  Warning: faithfulness eval failed: {e}")
        return 0.0


def run_eval(config_path: str = "config.yaml") -> dict:
    config = load_config(config_path)
    qa_data = load_golden_qa(config.eval.golden_qa_path)

    results = []
    for qa in qa_data:
        qid = qa["id"]
        question = qa["question"]
        expected_sources = qa.get("expected_sources", [])
        print(f"Evaluating {qid}: {question}")

        # Retrieval
        retrieved, recall = evaluate_retrieval(config, question, expected_sources)
        context_text = "\n\n".join(r.chunk.text for r in retrieved[:5])

        # Generation (if Ollama is available)
        answer_text = ""
        faith_score = 0.0
        cite_score = 0.0
        sources = []

        try:
            from ragged.pipeline import RAGPipeline

            pipeline = RAGPipeline(config)
            answer = pipeline.query(question)
            answer_text = answer.text
            sources = [s.model_dump() for s in answer.sources]

            # Faithfulness
            faith_score = evaluate_faithfulness(config, answer_text, context_text)

            # Citation accuracy
            cite_score = citation_accuracy(answer_text, sources, expected_sources)
        except Exception as e:
            print(f"  Warning: generation failed: {e}")

        results.append({
            "id": qid,
            "question": question,
            "retrieval_recall": recall,
            "faithfulness": faith_score,
            "citation_accuracy": cite_score,
        })

    # Summary
    avg_recall = sum(r["retrieval_recall"] for r in results) / len(results) if results else 0
    avg_faith = sum(r["faithfulness"] for r in results) / len(results) if results else 0
    avg_cite = sum(r["citation_accuracy"] for r in results) / len(results) if results else 0

    summary = {
        "results": results,
        "averages": {
            "retrieval_recall": avg_recall,
            "faithfulness": avg_faith,
            "citation_accuracy": avg_cite,
        },
        "thresholds": config.eval.thresholds.model_dump(),
    }

    # Print table
    print("\n" + "=" * 60)
    table = [[r["id"], f"{r['retrieval_recall']:.2f}", f"{r['faithfulness']:.2f}", f"{r['citation_accuracy']:.2f}"] for r in results]
    table.append(["AVG", f"{avg_recall:.2f}", f"{avg_faith:.2f}", f"{avg_cite:.2f}"])
    print(tabulate(table, headers=["ID", "Recall@K", "Faithfulness", "Citations"], tablefmt="grid"))

    # Check thresholds
    passed = True
    thresholds = config.eval.thresholds
    if avg_recall < thresholds.retrieval_recall_at_k:
        print(f"\nFAIL: retrieval_recall ({avg_recall:.2f}) < threshold ({thresholds.retrieval_recall_at_k})")
        passed = False
    if avg_faith < thresholds.faithfulness:
        print(f"\nFAIL: faithfulness ({avg_faith:.2f}) < threshold ({thresholds.faithfulness})")
        passed = False
    if avg_cite < thresholds.citation_accuracy:
        print(f"\nFAIL: citation_accuracy ({avg_cite:.2f}) < threshold ({thresholds.citation_accuracy})")
        passed = False

    if passed:
        print("\nPASSED: All metrics above thresholds.")

    # Save results
    results_path = Path("eval/results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return summary if passed else sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_eval(args.config)
