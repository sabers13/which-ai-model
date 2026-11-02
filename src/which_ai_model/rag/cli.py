"""RAG lane commands: build the docs corpus and index it in pgvector."""

from __future__ import annotations

from pathlib import Path

import typer

from which_ai_model.rag import corpus, embed, store

app = typer.Typer(name="rag", help="Docs corpus and pgvector index.", no_args_is_help=True)

#: Local pgvector started by compose.yaml (port 5433 -> 5432, database `wam`).
DEFAULT_DSN = "postgresql://wam:wam@localhost:5433/wam"


@app.command("index")
def index_command(
    guide: Path = typer.Option(Path("docs/guide"), help="Directory of guide Markdown pages."),
    dsn: str = typer.Option(DEFAULT_DSN, help="Postgres DSN of the pgvector service."),
    table: str = typer.Option("wam_chunks", help="Table to upsert passages into."),
    batch_size: int = typer.Option(32, help="Embedding batch size."),
) -> None:
    """Embed the guide corpus and upsert it into pgvector (needs Postgres)."""
    passages = corpus.load_corpus(guide)
    chunks = corpus.chunk_documents(passages)
    typer.echo(f"corpus: {len(passages)} passages -> {len(chunks)} chunks")
    vectors = embed.embed_texts([chunk.text for chunk in chunks], batch_size=batch_size)
    typer.echo(f"embed: {embed.EMBED_MODEL_NAME}@{embed.EMBED_MODEL_REVISION[:12]}")
    vector_store = store.PgVectorStore(dsn, table=table)
    vector_store.ensure_schema(dimensions=embed.EMBED_DIMENSIONS)
    vector_store.upsert(chunks, vectors)
    typer.echo(f"index: upserted {len(chunks)} chunks into table {table}")
