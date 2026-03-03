SYSTEM_PROMPT = """You are a helpful documentation assistant. Answer questions based ONLY on the provided context.

Rules:
1. Every factual claim MUST cite its source using [1], [2], etc. corresponding to the numbered context chunks below.
2. If the provided context does not contain enough information to answer the question, say: "I don't have enough information to answer this based on the available documentation."
3. Do NOT make up information or use knowledge outside the provided context.
4. After your answer, include a "Sources:" section that maps each citation number to the source document.

Format your response as:
<answer with inline citations [1], [2], etc.>

Sources:
[1] <source file and section>
[2] <source file and section>
..."""


def build_user_prompt(query: str, context_chunks: list[dict]) -> str:
    """Build the user prompt with numbered context chunks."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("source", "unknown")
        section = chunk.get("section", "")
        header = f"[{i}] Source: {source}"
        if section:
            header += f" | Section: {section}"
        context_parts.append(f"{header}\n{chunk['text']}")

    context_str = "\n\n---\n\n".join(context_parts)

    return f"""Context:
{context_str}

Question: {query}"""
