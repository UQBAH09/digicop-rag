"""
QdrantWriter — the only class that writes to Qdrant in this pipeline.

Orphan-cleanup contract: replace_book() always deletes all existing points
for a book before inserting the new set. After the call, the collection
contains exactly the points produced by the current chunker run for that
book, and no others. The orchestrator must always call replace_book() —
never raw upsert — to preserve this guarantee.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType, FilterSelector, Filter, FieldCondition, MatchValue, PointStruct
import uuid

from shared.config import qdrant_settings, QdrantSettings
from pipeline.embedding.base import BaseEmbedder
from shared.models import Chunk


class QdrantWriter:
    def __init__(self, embedder: BaseEmbedder, settings: QdrantSettings = qdrant_settings):
        self._settings = settings
        self._client = QdrantClient(url=self._settings.url)
        self._embedder = embedder

    def ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._settings.collection_name not in existing:
            self._client.create_collection(
                collection_name=self._settings.collection_name,
                vectors_config=VectorParams(size=self._embedder.dim, distance=Distance.COSINE)
            )

        for field in ("client_id", "book_id", "subject", "grade", "lang", "chapter", "element_type"):
            self._client.create_payload_index(
                collection_name=self._settings.collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
            )
    
    def _chunk_id_to_uuid(self, chunk_id: str) -> str:
        # Qdrant requires point IDs to be unsigned integers or UUIDs.
        # chunk_id is a SHA-256 hex string — take the first 32 chars to form a valid UUID.
        return str(uuid.UUID(chunk_id[:32]))

    def _build_payload(self, chunk: Chunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "client_id": chunk.client_id,
            "element_type": chunk.element_type,
            "book_id": chunk.book_id,
            "subject": chunk.subject,
            "grade": chunk.grade,
            "lang": chunk.lang,
            "chapter": chunk.chapter,
            "section": chunk.section,
        }

    def replace_book(self, book_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """
        Delete all existing points for book_id, then insert the new set.

        After this call the collection contains exactly the points produced
        by the current chunker run for this book — no orphans from prior runs.
        """
        self._client.delete(
            collection_name=self._settings.collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))])
            )
        )

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            points.append(PointStruct(
                id=self._chunk_id_to_uuid(chunk.chunk_id),
                vector=vector,
                payload=self._build_payload(chunk)
            ))

        for i in range(0, len(points), self._settings.upsert_batch_size):
            batch = points[i: i + self._settings.upsert_batch_size]
            self._client.upsert(
                collection_name=self._settings.collection_name,
                points=batch,
            )