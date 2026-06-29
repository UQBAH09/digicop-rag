import pytest
import hashlib
import uuid
from qdrant_client import QdrantClient
from shared.models import Chunk
from shared.config import QdrantSettings
from pipeline.embedding.base import BaseEmbedder
from pipeline.indexing.qdrant_writer import QdrantWriter
from tests.pipeline.test_embedder import FakeEmbedder


# --- Helpers ---

TEST_COLLECTION = "test_textbooks"


def make_settings() -> QdrantSettings:
    """Points to local Qdrant but uses a separate test collection."""
    return QdrantSettings(
        url="http://localhost:6333",
        collection_name=TEST_COLLECTION,
        upsert_batch_size=256,
    )


def make_chunk(book_id: str, chunk_id: str, content: str = "test content") -> Chunk:
    real_chunk_id = hashlib.sha256(chunk_id.encode()).hexdigest()
    return Chunk(
        chunk_id=real_chunk_id,
        chunk_index=0,
        content=content,
        element_type="text",
        book_id=book_id,
        subject="English",
        grade="9",
        board="Federal",
        lang="en",
        page_start=1,
        page_end=1,
    )


@pytest.fixture(autouse=True)
def clean_collection():
    """Delete the test collection before and after every test so tests don't bleed into each other."""
    client = QdrantClient(url="http://localhost:6333")
    # cleanup before
    existing = [c.name for c in client.get_collections().collections]
    if TEST_COLLECTION in existing:
        client.delete_collection(TEST_COLLECTION)
    yield
    # cleanup after
    existing = [c.name for c in client.get_collections().collections]
    if TEST_COLLECTION in existing:
        client.delete_collection(TEST_COLLECTION)

# --- Tests ---

def test_ensure_collection_creates_collection():
    embedder = FakeEmbedder()
    writer = QdrantWriter(embedder=embedder, settings=make_settings())
    writer.ensure_collection()

    client = QdrantClient(url="http://localhost:6333")
    existing = [c.name for c in client.get_collections().collections]
    assert TEST_COLLECTION in existing


def test_ensure_collection_is_idempotent():
    """Calling ensure_collection twice must not raise."""
    embedder = FakeEmbedder()
    writer = QdrantWriter(embedder=embedder, settings=make_settings())
    writer.ensure_collection()
    writer.ensure_collection()  # second call must be a no-op


def test_payload_indexes_created():
    """All filter fields must have payload indexes after ensure_collection."""
    embedder = FakeEmbedder()
    writer = QdrantWriter(embedder=embedder, settings=make_settings())
    writer.ensure_collection()

    client = QdrantClient(url="http://localhost:6333")
    info = client.get_collection(TEST_COLLECTION)
    indexed_fields = list(info.payload_schema.keys())

    for field in ("book_id", "subject", "grade", "lang", "chapter", "element_type"):
        assert field in indexed_fields, f"Missing payload index for field: {field}"


def test_replace_book_orphan_cleanup():
    """Insert 5 chunks, replace with 3 — collection must have exactly 3 for that book."""
    embedder = FakeEmbedder()
    writer = QdrantWriter(embedder=embedder, settings=make_settings())
    writer.ensure_collection()

    book_id = "book-001"

    # insert 5 chunks
    chunks_5 = [make_chunk(book_id, f"id-{i}", f"content {i}") for i in range(5)]
    vectors_5 = embedder.embed(chunks_5)
    writer.replace_book(book_id, chunks_5, vectors_5)

    # replace with 3 chunks
    chunks_3 = [make_chunk(book_id, f"id-new-{i}", f"new content {i}") for i in range(3)]
    vectors_3 = embedder.embed(chunks_3)
    writer.replace_book(book_id, chunks_3, vectors_3)

    # confirm exactly 3 points exist for this book
    client = QdrantClient(url="http://localhost:6333")
    result = client.count(
        collection_name=TEST_COLLECTION,
        count_filter={"must": [{"key": "book_id", "match": {"value": book_id}}]},
        exact=True,
    )
    assert result.count == 3


def test_payload_contains_all_fields():
    """Every field from _build_payload must be present on the stored point."""
    embedder = FakeEmbedder()
    writer = QdrantWriter(embedder=embedder, settings=make_settings())
    writer.ensure_collection()

    chunk = make_chunk("book-001", "chunk-001", "some content")
    vectors = embedder.embed([chunk])
    writer.replace_book("book-001", [chunk], vectors)

    client = QdrantClient(url="http://localhost:6333")
    point_uuid = str(uuid.UUID(chunk.chunk_id[:32]))
    results = client.retrieve(
        collection_name=TEST_COLLECTION,
        ids=[point_uuid],
        with_payload=True,
    )
    assert len(results) == 1
    payload = results[0].payload

    expected_fields = [
        "chunk_id", "chunk_index", "content", "element_type",
        "book_id", "subject", "grade", "board", "lang",
        "chapter", "section", "page_start", "page_end",
        "context_snippet", "image_path",
    ]
    for field in expected_fields:
        assert field in payload, f"Missing field in payload: {field}"