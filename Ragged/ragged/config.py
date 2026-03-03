from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    url: str = "http://localhost:11434"
    model: str = "llama3.2"


class EmbeddingsConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"


class RerankerConfig(BaseModel):
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64


class RetrievalConfig(BaseModel):
    top_k: int = 20
    rrf_k: int = 60


class StorageConfig(BaseModel):
    chroma_path: str = "./data/chroma"
    bm25_path: str = "./data/bm25"
    hash_db_path: str = "./data/file_hashes.db"


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class EvalThresholds(BaseModel):
    retrieval_recall_at_k: float = 0.8
    faithfulness: float = 0.7
    citation_accuracy: float = 0.8


class EvalConfig(BaseModel):
    golden_qa_path: str = "./eval/golden_qa.yaml"
    thresholds: EvalThresholds = Field(default_factory=EvalThresholds)


class RaggedConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


def load_config(path: str | Path = "config.yaml") -> RaggedConfig:
    """Load config from a YAML file. Falls back to defaults if file not found."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return RaggedConfig(**data)
    return RaggedConfig()
