import hashlib
import random

import pytest

from shared.models import Chunk
from pipeline.embedding.base import BaseEmbedder
from tests.pipeline.fake_embedder import FakeEmbedder


# --- Helper ---

def make_chunk(content: str) -> Chunk:
    """Creates a minimal valid Chunk for testing purposes."""
    return Chunk(
        chunk_id="test-id-001",
        chunk_index=0,
        content=content,
        element_type="text",
        book_id="book-001",
        subject="English",
        grade="9",
        board="Federal",
        lang="en",
        page_start=1,
        page_end=1,
    )


# --- Tests ---

def test_embed_returns_one_vector_per_chunk():
    embedder = FakeEmbedder()
    chunks = [make_chunk("Hello world"), make_chunk("Another chunk")]
    result = embedder.embed(chunks)
    assert len(result) == len(chunks)


def test_each_vector_has_correct_dimension():
    embedder = FakeEmbedder()
    chunks = [make_chunk("Some content")]
    result = embedder.embed(chunks)
    assert len(result[0]) == embedder.dim


def test_embedding_is_deterministic():
    embedder = FakeEmbedder()
    chunk = make_chunk("Determinism check")
    result1 = embedder.embed([chunk])
    result2 = embedder.embed([chunk])
    assert result1[0] == result2[0]