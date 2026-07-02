"""
PostgresWriter — writes books, chapters, sections, and chunks to Postgres.

Postgres is the source of truth for all content and metadata.
Qdrant holds only vectors and filter fields; it's rebuildable from Postgres.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import delete, select

from shared.models import Document, Chunk
from pipeline.persistence.models import Book, Client, Chapter, Section, ChunkRow


class PostgresWriter:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def ensure_client(self, client_id: str, name: str | None = None) -> None:
        """Create a Client row if it doesn't exist."""
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    select(Client).where(Client.client_id == client_id)
                )
                if result.scalar_one_or_none() is None:
                    session.add(
                        Client(
                            client_id=client_id,
                            name=name or client_id,
                        )
                    )

    async def replace_book(self, document: Document, chunks: list[Chunk]) -> None:
        """
        Atomically delete the book's existing rows (chapters, sections, chunks
        via CASCADE) and insert the fresh set. Idempotent.
        """
        meta = document.meta

        async with self._session_factory() as session:
            async with session.begin():
                # 1. Delete existing book — CASCADE removes chapters, sections, chunks
                await session.execute(delete(Book).where(Book.book_id == meta.book_id))

                # 2. Insert book row
                book = Book(
                    book_id=meta.book_id,
                    client_id=meta.client_id,
                    title=meta.title,
                    subject=meta.subject,
                    grade=meta.grade,
                    board=meta.board,
                    lang=meta.lang,
                    source_path=document.source_path,
                    extractor_name=document.extractor_name,
                )
                session.add(book)

                # 3. Extract distinct chapters in order of first appearance
                chapter_map: dict[str, Chapter] = {}
                for chunk in chunks:
                    if chunk.chapter is not None and chunk.chapter not in chapter_map:
                        ch = Chapter(
                            book_id=meta.book_id,
                            name=chunk.chapter,
                            order_index=len(chapter_map),
                        )
                        session.add(ch)
                        chapter_map[chunk.chapter] = ch

                # Flush to populate auto-generated chapter_id values
                await session.flush()

                # 4. Extract distinct sections in order of first appearance
                section_map: dict[tuple[int | None, str], Section] = {}
                for chunk in chunks:
                    if chunk.section is not None:
                        # Section key includes chapter_id to handle same section
                        # name under different chapters
                        ch_id = (
                            chapter_map[chunk.chapter].chapter_id
                            if chunk.chapter
                            else None
                        )
                        key = (ch_id, chunk.section)
                        if key not in section_map:
                            sec = Section(
                                book_id=meta.book_id,
                                chapter_id=ch_id,
                                name=chunk.section,
                                order_index=len(section_map),
                            )
                            session.add(sec)
                            section_map[key] = sec

                # Flush to populate auto-generated section_id values
                await session.flush()

                # 5. Bulk insert chunk rows with resolved FKs
                for chunk in chunks:
                    ch_id = None
                    sec_id = None

                    if chunk.chapter is not None:
                        ch_id = chapter_map[chunk.chapter].chapter_id

                    if chunk.section is not None:
                        sec_ch_id = ch_id  # section is under this chapter
                        key = (sec_ch_id, chunk.section)
                        sec_id = section_map[key].section_id

                    session.add(
                        ChunkRow(
                            chunk_id=chunk.chunk_id,
                            book_id=meta.book_id,
                            client_id=meta.client_id,
                            chunk_index=chunk.chunk_index,
                            content=chunk.content,
                            element_type=chunk.element_type,
                            chapter_id=ch_id,
                            section_id=sec_id,
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            context_snippet=chunk.context_snippet,
                            image_path=chunk.image_path,
                            lang=chunk.lang,
                        )
                    )
