from shared.models import Document, Chunk
from pipeline.embedding.base import BaseEmbedder
from pipeline.indexing.qdrant_writer import QdrantWriter
from pipeline.persistence.postgres_writer import PostgresWriter


async def ingest_book(
    document: Document,
    chunks: list[Chunk],
    postgres_writer: PostgresWriter,
    embedder: BaseEmbedder,
    qdrant_writer: QdrantWriter,
) -> None:
    # 1. ensure client exists, then persist to Postgres
    postgres_writer.ensure_client(client_id=document.meta.client_id)
    postgres_writer.replace_book(document=document, chunks=chunks)

    # 2. embed chunks
    vectors = embedder.embed(chunks=chunks)

    # 3. write to Qdrant
    qdrant_writer.replace_book(book_id=document.meta.book_id, chunks=chunks, vectors=vectors)