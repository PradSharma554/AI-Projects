# Ragged

A fully local, domain-specific RAG (Retrieval-Augmented Generation) system with hybrid retrieval, cross-encoder reranking, citation enforcement, and a CI-gated evaluation pipeline. Runs entirely on local hardware with zero cloud dependencies.

## Features

- **Hybrid Retrieval** — BM25 keyword search + ChromaDB vector search merged via Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking** — Reranks candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` for precision
- **Citation Enforcement** — Every claim in the answer cites its source as `[1]`, `[2]`, etc.
- **Multi-Format Ingestion** — Parses Markdown (with frontmatter), PDF, HTML, DOCX, and plain text
- **Incremental Indexing** — Tracks file hashes to skip unchanged documents on re-ingestion
- **Fully Local** — Ollama for LLM inference, sentence-transformers for embeddings, no API keys needed
- **Three Interfaces** — REST API, CLI tool, and web chat UI
- **Evaluation Pipeline** — CI-gated metrics for retrieval recall, faithfulness, and citation accuracy

## Quickstart

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

Pull a model:

```bash
ollama pull llama3.2
```

### Install

```bash
git clone <repo-url> && cd Ragged
python -m venv .venv
source .venv/bin/activate
pip install .
```

For development dependencies (pytest, ruff):

```bash
pip install ".[dev]"
```

### Ingest Documents

```bash
ragged ingest ./path/to/docs
```

### Ask a Question

```bash
ragged ask "How do I configure authentication?"
```

### Start the API Server

```bash
ragged serve
```

The web UI is available at `http://localhost:8000` and the API at `http://localhost:8000/api/`.

## Architecture

```
Documents (MD/PDF/HTML/DOCX/TXT)
        |
        v
  Ingestion Pipeline --- Parse, chunk, extract metadata
        |
   +---------+
   v         v
ChromaDB   BM25 Index
(vectors)  (keyword)
   |         |
   +----+----+
        |
        v  <-- User query
  Hybrid Retrieval --- BM25 + vector search, RRF merge
        |
        v
  Cross-Encoder Reranker --- Rerank top-K candidates
        |
        v
  Answer Generator --- Ollama LLM + citation-enforcing prompt
        |
        v
  Answer + [1][2]... Sources
```

## CLI Usage

```
ragged ingest <directory>   Ingest documents from a directory
ragged ask "<question>"     Ask a question, get a cited answer
ragged status               Show index statistics (collections, chunk counts)
ragged serve                Start the FastAPI server
```

Options available on most commands:

| Flag | Default | Description |
|------|---------|-------------|
| `--collection` | `default` | Target collection name |
| `--config` | `config.yaml` | Path to config file |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/query` | Ask a question, get answer with citations |
| `POST` | `/api/ingest` | Ingest documents from a directory path |
| `GET` | `/api/health` | Health check (Ollama status, collection count) |
| `GET` | `/api/collections` | List all indexed collections |
| `DELETE` | `/api/collections/{name}` | Delete a collection |

### Query Example

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I configure auth?", "collection": "default"}'
```

### Ingest Example

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "./docs", "collection": "default"}'
```

## Web UI

The chat interface is served at `http://localhost:8000` when the API server is running. It provides:

- Chat-like question/answer interface
- Inline citations (`[1]`, `[2]`, ...) in answers
- Expandable source panels showing chunk text
- Collection management page at `/collections`

## Configuration

All settings live in `config.yaml`:

```yaml
ollama:
  url: "http://localhost:11434"
  model: "llama3.2"

embeddings:
  model: "all-MiniLM-L6-v2"

reranker:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_n: 5

chunking:
  chunk_size: 512    # tokens
  chunk_overlap: 64  # tokens

retrieval:
  top_k: 20   # candidates from hybrid retrieval
  rrf_k: 60   # RRF constant

storage:
  chroma_path: "./data/chroma"
  bm25_path: "./data/bm25"
  hash_db_path: "./data/file_hashes.db"

api:
  host: "0.0.0.0"
  port: 8000
```

## Docker

Run the full stack (Ollama + Ragged) with Docker Compose:

```bash
docker compose up --build
```

This starts:
- **Ollama** on port `11434`
- **Ragged API + Web UI** on port `8000`

Mount your documents into the `./docs` volume and persistent data is stored in `./data`.

## Evaluation

The evaluation pipeline measures retrieval and generation quality against a golden QA dataset.

### Metrics

| Metric | Description |
|--------|-------------|
| **Retrieval Recall@K** | Fraction of expected source documents found in top-K retrieved chunks |
| **Faithfulness** | LLM-as-judge score (0-1): are all claims supported by the provided context? |
| **Citation Accuracy** | Fraction of citations that correctly map to relevant source chunks |

### Run Locally

```bash
make eval
```

### Golden QA Dataset

Define ground-truth questions in `eval/golden_qa.yaml`:

```yaml
questions:
  - id: "q1"
    question: "How do I install Ragged?"
    expected_sources: ["sample.md"]
    expected_keywords: ["pip", "install"]
    reference_answer: "Install Ragged with pip install ."
```

### CI Integration

A GitHub Actions workflow (`.github/workflows/eval.yml`) runs the evaluation on every pull request to `main`. The pipeline fails if any metric drops below the thresholds configured in `config.yaml`:

```yaml
eval:
  thresholds:
    retrieval_recall_at_k: 0.8
    faithfulness: 0.7
    citation_accuracy: 0.8
```

## Project Structure

```
Ragged/
├── ragged/
│   ├── models.py              # Pydantic data models
│   ├── config.py              # YAML config loader
│   ├── cli.py                 # Typer CLI commands
│   ├── chunker.py             # Header/code-aware chunking
│   ├── indexer.py             # Ingestion orchestrator (ChromaDB + BM25)
│   ├── reranker.py            # Cross-encoder reranking
│   ├── generator.py           # Ollama answer generation with citations
│   ├── pipeline.py            # Full RAG pipeline
│   ├── prompts.py             # System/user prompt templates
│   ├── parsers/
│   │   ├── base.py            # BaseParser protocol
│   │   ├── markdown.py        # MD + frontmatter
│   │   ├── pdf.py             # PyMuPDF
│   │   ├── html.py            # BeautifulSoup
│   │   ├── docx.py            # python-docx
│   │   └── text.py            # Plain text
│   ├── retriever/
│   │   ├── bm25.py            # BM25 keyword search
│   │   ├── vector.py          # ChromaDB vector search
│   │   └── hybrid.py          # Reciprocal Rank Fusion
│   └── api/
│       ├── app.py             # FastAPI app factory
│       ├── routes.py          # API endpoints
│       └── templates/         # Jinja2 templates (chat, collections)
├── eval/
│   ├── runner.py              # Evaluation runner
│   ├── metrics.py             # Metric computations
│   └── golden_qa.yaml         # Ground-truth QA pairs
├── tests/                     # pytest test suite
├── config.yaml                # Default configuration
├── pyproject.toml             # Dependencies and project metadata
├── Dockerfile
├── docker-compose.yml
└── Makefile                   # Dev shortcuts
```

## Development

```bash
make dev        # Start server with hot reload
make test       # Run tests
make test-cov   # Run tests with coverage
make lint       # Lint with ruff
make format     # Format with ruff
make clean      # Remove generated data
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (local) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector store | ChromaDB |
| Keyword search | rank-bm25 |
| API | FastAPI + Uvicorn |
| CLI | Typer |
| Document parsing | PyMuPDF, BeautifulSoup, python-docx |
| CI | GitHub Actions |
