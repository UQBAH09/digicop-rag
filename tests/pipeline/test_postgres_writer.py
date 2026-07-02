"""Tests for PostgresWriter against a real local Postgres."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, func

from pipeline.persistence.models import Base, Client, Book, Chapter, Section, ChunkRow
from pipeline.persistence.postgres_writer import PostgresWriter
from shared.models import Chunk, Document, BookMeta, Page

TEST_DB_URL = "postgresql+asyncpg://digicop:digicop@localhost/digicop"


@pytest_asyncio.fixture
async def session_factory():
    """Create tables before each test, drop them after."""
    engine = create_async_engine(TEST_DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def writer(session_factory):
    return PostgresWriter(session_factory)


def make_document(book_id: str = "book-001", client_id: str = "client-001") -> Document:
    return Document(
        meta=BookMeta(
            book_id=book_id,
            client_id=client_id,
            title="Physics XI",
            subject="Science",
            grade="11",
            board="Federal",
            lang="en",
        ),
        pages=[],
        source_path="/data/physics.pdf",
        extractor_name="docling",
    )


def make_chunks(
    book_id: str = "book-001", client_id: str = "client-001", count: int = 5
) -> list[Chunk]:
    chapters = ["Motion", "Motion", "Forces", "Forces", "Energy"]
    sections = ["Velocity", "Acceleration", "Gravity", "Friction", "Kinetic"]
    return [
        Chunk(
            chunk_id=f"chunk-{book_id}-{i}",
            chunk_index=i,
            content=f"Content for chunk {i}",
            element_type="text",
            book_id=book_id,
            client_id=client_id,
            subject="Science",
            grade="11",
            board="Federal",
            lang="en",
            chapter=chapters[i] if i < len(chapters) else None,
            section=sections[i] if i < len(sections) else None,
            page_start=i + 1,
            page_end=i + 1,
        )
        for i in range(count)
    ]


# --- Tests ---


@pytest.mark.asyncio
async def test_ensure_client_creates_client(writer, session_factory):
    await writer.ensure_client("client-001", "Test School")

    async with session_factory() as session:
        result = await session.execute(
            select(Client).where(Client.client_id == "client-001")
        )
        client = result.scalar_one()
        assert client.name == "Test School"


@pytest.mark.asyncio
async def test_ensure_client_is_idempotent(writer):
    """Calling ensure_client twice must not raise."""
    await writer.ensure_client("client-001", "Test School")
    await writer.ensure_client("client-001", "Test School")


@pytest.mark.asyncio
async def test_replace_book_inserts_book_and_chunks(writer, session_factory):
    doc = make_document()
    chunks = make_chunks(count=5)

    await writer.ensure_client("client-001")
    await writer.replace_book(doc, chunks)

    async with session_factory() as session:
        book_count = await session.scalar(select(func.count()).select_from(Book))
        chunk_count = await session.scalar(select(func.count()).select_from(ChunkRow))
        assert book_count == 1
        assert chunk_count == 5


@pytest.mark.asyncio
async def test_replace_book_populates_chapters(writer, session_factory):
    doc = make_document()
    chunks = make_chunks(count=5)  # has chapters: Motion, Forces, Energy

    await writer.ensure_client("client-001")
    await writer.replace_book(doc, chunks)

    async with session_factory() as session:
        result = await session.execute(
            select(Chapter)
            .where(Chapter.book_id == "book-001")
            .order_by(Chapter.order_index)
        )
        chapters = result.scalars().all()
        assert len(chapters) == 3
        assert [c.name for c in chapters] == ["Motion", "Forces", "Energy"]


@pytest.mark.asyncio
async def test_replace_book_resolves_chunk_chapter_fk(writer, session_factory):
    doc = make_document()
    chunks = make_chunks(count=2)  # both have chapter "Motion"

    await writer.ensure_client("client-001")
    await writer.replace_book(doc, chunks)

    async with session_factory() as session:
        # Get the chapter ID for "Motion"
        ch_result = await session.execute(
            select(Chapter).where(Chapter.name == "Motion")
        )
        chapter = ch_result.scalar_one()

        # Both chunks should reference this chapter
        chunk_result = await session.execute(
            select(ChunkRow).where(ChunkRow.book_id == "book-001")
        )
        chunk_rows = chunk_result.scalars().all()
        for row in chunk_rows:
            assert row.chapter_id == chapter.chapter_id


@pytest.mark.asyncio
async def test_replace_book_is_atomic(writer, session_factory):
    """Insert 5 chunks, replace with 3 — exactly 3 remain."""
    doc = make_document()

    await writer.ensure_client("client-001")
    await writer.replace_book(doc, make_chunks(count=5))
    await writer.replace_book(doc, make_chunks(count=3))

    async with session_factory() as session:
        chunk_count = await session.scalar(
            select(func.count())
            .select_from(ChunkRow)
            .where(ChunkRow.book_id == "book-001")
        )
        assert chunk_count == 3


@pytest.mark.asyncio
async def test_replace_book_cascades_chapters(writer, session_factory):
    """When a book is replaced, old chapters are deleted via CASCADE."""
    doc = make_document()

    await writer.ensure_client("client-001")
    await writer.replace_book(doc, make_chunks(count=5))  # 3 chapters

    # Replace with chunks that have only 1 chapter
    new_chunks = [
        Chunk(
            chunk_id=f"chunk-new-{i}",
            chunk_index=i,
            content=f"New content {i}",
            element_type="text",
            book_id="book-001",
            client_id="client-001",
            subject="Science",
            grade="11",
            board="Federal",
            lang="en",
            chapter="Waves",
            section=None,
            page_start=1,
            page_end=1,
        )
        for i in range(2)
    ]
    await writer.replace_book(doc, new_chunks)

    async with session_factory() as session:
        ch_count = await session.scalar(
            select(func.count())
            .select_from(Chapter)
            .where(Chapter.book_id == "book-001")
        )
        assert ch_count == 1
