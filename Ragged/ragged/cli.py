from __future__ import annotations

from pathlib import Path

import typer

from ragged.config import load_config

app = typer.Typer(name="ragged", help="Ask My Docs — local RAG with citations")


@app.command()
def ingest(
    directory: str = typer.Argument(..., help="Path to the documents directory"),
    collection: str = typer.Option("default", help="Collection name"),
    config: str = typer.Option("config.yaml", help="Path to config file"),
):
    """Ingest documents from a directory into the index."""
    cfg = load_config(config)
    from ragged.indexer import Indexer

    indexer = Indexer(cfg)
    path = Path(directory)
    if not path.is_dir():
        typer.echo(f"Error: {directory} is not a valid directory", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Ingesting documents from {path} into collection '{collection}'...")
    result = indexer.ingest(path, collection=collection)
    typer.echo(f"Done: {result.files_processed} files processed, {result.chunks_created} chunks created, {result.files_skipped} files skipped")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask"),
    collection: str = typer.Option("default", help="Collection to query"),
    config: str = typer.Option("config.yaml", help="Path to config file"),
):
    """Ask a question and get an answer with citations."""
    cfg = load_config(config)
    from ragged.pipeline import RAGPipeline

    pipeline = RAGPipeline(cfg)
    typer.echo(f"Querying: {question}\n")
    answer = pipeline.query(question, collection=collection)
    typer.echo(answer.text)

    if answer.sources:
        typer.echo("\n--- Sources ---")
        for src in answer.sources:
            section_str = f" | {src.section}" if src.section else ""
            page_str = f" | page {src.page}" if src.page else ""
            typer.echo(f"[{src.index}] {src.source}{section_str}{page_str}")


@app.command()
def status(
    config: str = typer.Option("config.yaml", help="Path to config file"),
):
    """Show index statistics."""
    cfg = load_config(config)
    from ragged.indexer import Indexer

    indexer = Indexer(cfg)
    collections = indexer.list_collections()

    if not collections:
        typer.echo("No collections indexed yet.")
        return

    typer.echo(f"Collections: {len(collections)}\n")
    for col in collections:
        typer.echo(f"  {col.name}: {col.chunk_count} chunks")


@app.command()
def serve(
    config: str = typer.Option("config.yaml", help="Path to config file"),
):
    """Start the API server."""
    import uvicorn

    cfg = load_config(config)
    uvicorn.run(
        "ragged.api.app:create_app",
        host=cfg.api.host,
        port=cfg.api.port,
        factory=True,
    )


if __name__ == "__main__":
    app()
