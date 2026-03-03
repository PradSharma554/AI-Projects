from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ragged.config import load_config
from ragged.models import Answer, CollectionInfo, IngestResult
from ragged.pipeline import RAGPipeline

router = APIRouter()

_config = load_config()
_pipeline = RAGPipeline(_config)


class QueryRequest(BaseModel):
    question: str
    collection: str = "default"


class IngestRequest(BaseModel):
    directory: str
    collection: str = "default"


@router.post("/query", response_model=Answer)
def query(req: QueryRequest):
    """Ask a question and get an answer with citations."""
    try:
        return _pipeline.query(req.question, collection=req.collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestResult)
def ingest(req: IngestRequest):
    """Ingest documents from a directory."""
    path = Path(req.directory)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a valid directory: {req.directory}")
    try:
        return _pipeline.indexer.ingest(path, collection=req.collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health():
    """Health check."""
    import httpx

    ollama_ok = False
    try:
        resp = httpx.get(f"{_config.ollama.url}/api/tags", timeout=5.0)
        ollama_ok = resp.status_code == 200
    except Exception:
        pass

    collections = _pipeline.indexer.list_collections()
    return {
        "status": "ok",
        "ollama": "connected" if ollama_ok else "unavailable",
        "collections": len(collections),
    }


@router.get("/collections", response_model=list[CollectionInfo])
def list_collections():
    """List all indexed collections."""
    return _pipeline.indexer.list_collections()


@router.delete("/collections/{name}")
def delete_collection(name: str):
    """Delete a collection."""
    try:
        _pipeline.indexer.delete_collection(name)
        return {"status": "deleted", "collection": name}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
