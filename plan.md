# Ragged — "Ask My Docs" RAG System

<!-- Press Cmd+Shift+V to preview -->

## Context

Build a fully local, domain-specific RAG system with hybrid retrieval, cross-encoder reranking, citation enforcement, and a CI-gated evaluation pipeline. This is the most common enterprise AI pattern — adapted here to run entirely on local hardware with zero cloud dependencies.

**Project location**: `/Users/zopdev/Desktop/This PC/Coding/AI/Ragged`

---

## Technology Decisions (Finalized)

| Component      | Choice                              | Details                                                    |
| -------------- | ----------------------------------- | ---------------------------------------------------------- |
| Language       | Python                              | Core language for all modules                              |
| LLM            | Ollama (local)                      | Fully local, no API keys, no cloud                         |
| Embeddings     | sentence-transformers               | Local models (e.g., all-MiniLM-L6-v2 or bge-small-en-v1.5) |
| Reranker       | Local cross-encoder                 | cross-encoder/ms-marco-MiniLM-L-6-v2                       |
| Vector store   | ChromaDB                            | Persistent, embedded mode                                  |
| BM25           | rank_bm25                           | Keyword search component of hybrid retrieval               |
| Document types | MD, PDF, HTML, DOCX, TXT            | All common formats                                         |
| API framework  | FastAPI                             | REST API layer                                             |
| CLI            | Click or Typer                      | Command-line interface                                     |
| Web UI         | Jinja2 templates + HTMX or plain JS | Served by FastAPI, simple chat interface                   |
| CI             | GitHub Actions                      | Evaluation pipeline gated on PRs                           |
| Config         | YAML                                | Single config file for all settings                        |

---

## Core Architecture

### Data Flow

```
Documents (MD/PDF/HTML/DOCX/TXT)
        │
        ▼
┌─────────────────┐
│  Ingestion       │  ← Parse, chunk, extract metadata
│  Pipeline        │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
 ChromaDB   BM25 Index
 (vectors)  (keyword)
    │         │
    └────┬────┘
         │
         ▼  ← User query
┌─────────────────┐
│  Hybrid Retrieval│  ← BM25 + vector search → RRF merge
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Cross-Encoder   │  ← Rerank top-K candidates
│  Reranker        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Answer Generator│  ← Ollama LLM + citation-enforcing prompt
│  (with citations)│
└─────────────────┘
         │
         ▼
   Answer + [1][2]... Sources
```

### Interfaces

```
                ┌──────────┐
                │  FastAPI  │ ← REST API
                │  Server   │
                ├──────────┤
  CLI ─────────►│ /query   │◄──────── Web UI
  (ragged ask)  │ /ingest  │         (chat interface)
                │ /health  │
                │ /collections│
                └──────────┘
```

---

## Feature Requirements

### 1. Document Ingestion Pipeline

- Parse all formats: MD (with frontmatter), PDF (PyMuPDF), HTML (BeautifulSoup), DOCX (python-docx), TXT
- Intelligent chunking: respect headers, paragraphs, code blocks; configurable chunk size + overlap
- Store in ChromaDB with embeddings + metadata (source file, section title, page number, chunk index)
- Build parallel BM25 index from same chunks
- Incremental ingestion: detect changed/new/deleted files via file hash tracking

### 2. Hybrid Retrieval (BM25 + Vector Search)

- Run both BM25 keyword search and ChromaDB vector similarity in parallel
- Merge results using Reciprocal Rank Fusion (RRF) with configurable weights
- Return top-K candidate chunks

### 3. Cross-Encoder Reranking

- Score each (query, chunk) pair with cross-encoder/ms-marco-MiniLM-L-6-v2
- Return top-N most relevant chunks after reranking
- Configurable N (default: 5)

### 4. Answer Generation with Citation Enforcement

- System prompt enforces: every claim must cite source using [1], [2], etc.
- Response includes "Sources" section mapping numbers → document paths/sections
- If no relevant context found → "I don't have enough information to answer this"
- Configurable Ollama model (default: llama3 or mistral)

### 5. REST API (FastAPI)

- `POST /query` — ask a question, get answer with citations
- `POST /ingest` — trigger ingestion from a directory path
- `GET /health` — health check (Ollama status, index stats)
- `GET /collections` — list indexed collections
- `DELETE /collections/{name}` — delete a collection

### 6. CLI Tool

- `ragged ingest <path>` — ingest documents from a directory
- `ragged ask "<question>"` — ask a question, get cited answer
- `ragged status` — show index statistics (doc count, chunk count, collections)
- Can target local API server or run in-process

### 7. Web UI

- Chat-like interface for asking questions
- Inline citations in answers ([1], [2], etc.)
- Expandable source panels showing actual chunk text
- Collection/index management page
- Served by FastAPI (Jinja2 templates + vanilla JS or HTMX)

### 8. CI-Gated Evaluation Pipeline (GitHub Actions)

- Golden QA dataset: questions + expected answers + expected source docs (YAML file)
- On PR, run full RAG pipeline for each question
- Metrics: retrieval recall@K, answer faithfulness (LLM-as-judge via Ollama), citation accuracy
- Fail CI if metrics drop below configurable thresholds
- Output eval results as GitHub Actions artifact + PR comment

---

## Project Structure

```
Ragged/
├── ragged/
│   ├── __init__.py
│   ├── config.py                 # Load & validate config.yaml
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseParser protocol
│   │   ├── markdown.py           # MD + frontmatter
│   │   ├── pdf.py                # PyMuPDF
│   │   ├── html.py               # BeautifulSoup
│   │   ├── docx.py               # python-docx
│   │   └── text.py               # Plain text
│   ├── chunker.py                # Intelligent chunking (header/paragraph/code-aware)
│   ├── indexer.py                 # Orchestrates ingestion → ChromaDB + BM25
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── bm25.py               # BM25 search
│   │   ├── vector.py             # ChromaDB vector search
│   │   └── hybrid.py             # RRF fusion of BM25 + vector
│   ├── reranker.py               # Cross-encoder reranking
│   ├── generator.py              # Ollama LLM answer generation
│   ├── prompts.py                # System/user prompt templates
│   ├── pipeline.py               # Full query pipeline: retrieve → rerank → generate
│   ├── models.py                 # Pydantic models (Chunk, Query, Answer, Source)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                # FastAPI app factory
│   │   ├── routes.py             # API endpoints
│   │   └── templates/            # Jinja2 templates for Web UI
│   │       ├── base.html
│   │       ├── chat.html
│   │       └── collections.html
│   ├── static/                   # CSS/JS for Web UI
│   │   ├── style.css
│   │   └── app.js
│   └── cli.py                    # Typer CLI commands
├── eval/
│   ├── golden_qa.yaml            # Golden QA dataset
│   ├── runner.py                 # Eval runner script
│   └── metrics.py                # Recall@K, faithfulness, citation accuracy
├── tests/
│   ├── test_parsers.py
│   ├── test_chunker.py
│   ├── test_indexer.py
│   ├── test_retriever.py
│   ├── test_reranker.py
│   ├── test_generator.py
│   ├── test_pipeline.py
│   ├── test_api.py
│   └── fixtures/                 # Sample docs for testing
│       ├── sample.md
│       ├── sample.pdf
│       ├── sample.html
│       ├── sample.docx
│       └── sample.txt
├── config.yaml                   # Default configuration
├── pyproject.toml                # Project metadata + dependencies
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── eval.yml              # CI evaluation pipeline
├── Makefile                      # Dev shortcuts
└── README.md
```

---

## Module Interfaces

### Parsers (`ragged/parsers/`)

```python
# base.py
class BaseParser(Protocol):
    def parse(self, file_path: Path) -> list[Document]
        """Return list of Document(text, metadata) from a file."""

# Each parser implements this. Metadata includes:
#   source: str (file path), page: int | None, section: str | None
```

A registry maps file extensions → parser classes. The indexer calls `get_parser(ext).parse(path)`.

### Chunker (`ragged/chunker.py`)

```python
def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split documents into chunks respecting headers, paragraphs, code blocks.
    Each Chunk carries forward the parent Document metadata + chunk_index."""
```

Strategy: split on `\n\n` (paragraphs) and `#` headers first, then apply token-level splitting if a section exceeds `chunk_size`. Code blocks (``` fenced) are kept whole if under 2x chunk_size.

### Indexer (`ragged/indexer.py`)

```python
class Indexer:
    def __init__(self, config: RaggedConfig): ...
    def ingest(self, directory: Path, collection: str = "default") -> IngestResult:
        """Parse all docs in directory, chunk, embed, store in ChromaDB + BM25.
        Returns stats: files_processed, chunks_created, skipped (unchanged)."""
    def delete_collection(self, collection: str) -> None: ...
    def list_collections(self) -> list[CollectionInfo]: ...
```

Tracks file hashes in a SQLite sidecar DB for incremental ingestion.

### Retriever (`ragged/retriever/`)

```python
# bm25.py
class BM25Retriever:
    def __init__(self, index_path: Path): ...
    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]: ...

# vector.py
class VectorRetriever:
    def __init__(self, chroma_client, collection: str, embedder): ...
    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]: ...

# hybrid.py
class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever, rrf_k: int = 60): ...
    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """Run both retrievers, merge with Reciprocal Rank Fusion."""
```

RRF formula: `score(d) = Σ 1 / (k + rank_i(d))` across both result lists.

### Reranker (`ragged/reranker.py`)

```python
class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"): ...
    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int = 5) -> list[ScoredChunk]:
        """Score each (query, chunk) pair, return top_n sorted by relevance."""
```

### Generator (`ragged/generator.py`)

```python
class Generator:
    def __init__(self, ollama_url: str, model: str): ...
    def generate(self, query: str, context_chunks: list[ScoredChunk]) -> Answer:
        """Build prompt with numbered context chunks, call Ollama, parse response.
        Returns Answer with text, citations list, and source mappings."""
```

### Pipeline (`ragged/pipeline.py`)

```python
class RAGPipeline:
    def __init__(self, retriever, reranker, generator): ...
    def query(self, question: str, collection: str = "default") -> Answer:
        """Full pipeline: retrieve → rerank → generate with citations."""
```

---

## Configuration (`config.yaml`)

```yaml
# Ollama
ollama:
  url: "http://localhost:11434"
  model: "llama3.2"

# Embeddings
embeddings:
  model: "all-MiniLM-L6-v2"

# Reranker
reranker:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_n: 5

# Chunking
chunking:
  chunk_size: 512 # tokens
  chunk_overlap: 64 # tokens

# Retrieval
retrieval:
  top_k: 20 # candidates from hybrid retrieval
  rrf_k: 60 # RRF constant
  bm25_weight: 1.0
  vector_weight: 1.0

# Storage
storage:
  chroma_path: "./data/chroma"
  bm25_path: "./data/bm25"
  hash_db_path: "./data/file_hashes.db"

# API
api:
  host: "0.0.0.0"
  port: 8000

# Evaluation
eval:
  golden_qa_path: "./eval/golden_qa.yaml"
  thresholds:
    retrieval_recall_at_k: 0.8
    faithfulness: 0.7
    citation_accuracy: 0.8
```

---

## Dependencies

```
# Core
fastapi              # API framework
uvicorn              # ASGI server
typer                # CLI framework
pydantic             # Data models
pyyaml               # Config parsing
jinja2               # Web UI templates
httpx                # HTTP client (Ollama API calls)

# Parsing
pymupdf              # PDF parsing
beautifulsoup4       # HTML parsing
python-docx          # DOCX parsing
python-frontmatter   # Markdown frontmatter

# AI/ML
sentence-transformers  # Local embeddings
chromadb               # Vector store
rank-bm25              # BM25 keyword search
torch                  # Required by sentence-transformers

# Evaluation
tabulate             # Pretty-print eval results

# Dev/Test
pytest
pytest-asyncio
pytest-cov
ruff                 # Linter + formatter
```

---

## Docker Setup

**docker-compose.yml**:

```yaml
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  ragged:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./docs:/app/docs # Mount your documents here
      - ./data:/app/data # ChromaDB + BM25 persistence
    environment:
      - OLLAMA_URL=http://ollama:11434
    depends_on:
      - ollama

volumes:
  ollama_data:
```

**Dockerfile**:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install .
COPY . .
CMD ["uvicorn", "ragged.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Evaluation Pipeline

### Golden QA Dataset Format (`eval/golden_qa.yaml`)

```yaml
questions:
  - id: "q1"
    question: "How do I configure authentication?"
    expected_sources: ["docs/auth.md", "docs/config.md"]
    expected_keywords: ["JWT", "token", "bearer"]
    reference_answer: "Authentication is configured via JWT tokens..."

  - id: "q2"
    question: "What databases are supported?"
    ...
```

### Metrics

| Metric                 | How Computed                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| **Retrieval Recall@K** | % of expected_sources that appear in top-K retrieved chunks                               |
| **Faithfulness**       | LLM-as-judge: "Is every claim in the answer supported by the provided context?" Score 0-1 |
| **Citation Accuracy**  | % of citations [N] that correctly map to a relevant source chunk                          |

### GitHub Actions Workflow (`.github/workflows/eval.yml`)

```yaml
name: RAG Evaluation
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    services:
      ollama:
        image: ollama/ollama
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install .
      - run: ollama pull llama3.2
      - run: python eval/runner.py --config config.yaml
      - uses: actions/upload-artifact@v4
        with: { name: eval-results, path: eval/results.json }
```

The eval runner exits with code 1 if any metric falls below its threshold, failing the CI check.

---

## Implementation Order

### Phase 1 — Core Foundation

1. Project scaffolding (pyproject.toml, config.yaml, models.py, config.py)
2. Parsers (all 5 formats)
3. Chunker
4. Unit tests for parsers + chunker

### Phase 2 — Indexing & Retrieval

5. ChromaDB vector indexer + embeddings
6. BM25 indexer
7. Vector retriever, BM25 retriever, Hybrid retriever (RRF)
8. Tests for retrieval

### Phase 3 — Reranking & Generation

9. Cross-encoder reranker
10. Ollama generator with citation-enforcing prompts
11. Full RAG pipeline (pipeline.py)
12. Tests for pipeline end-to-end

### Phase 4 — Interfaces

13. FastAPI REST API
14. CLI tool (Typer)
15. Web UI (Jinja2 templates + JS)

### Phase 5 — Evaluation & CI

16. Eval metrics + runner
17. Golden QA dataset (seed with sample docs)
18. GitHub Actions workflow

### Phase 6 — Docker & Polish

19. Dockerfile + docker-compose.yml
20. Makefile shortcuts
21. README
