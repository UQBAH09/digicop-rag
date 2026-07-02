"""
SQLAlchemy declarative models for the Postgres source-of-truth layer.

Five tables: clients, books, chapters, sections, chunks.
Vectors live in Qdrant, not here. This is the authoritative store
for all book content and metadata.
"""

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import ForeignKey, UniqueConstraint, Text, func
from datetime import datetime
class Base(DeclarativeBase):
    pass

class Client(Base):
    __tablename__ = "clients"
    client_id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class Book(Base):
    __tablename__ = "books"
    book_id: Mapped[str] = mapped_column(primary_key=True)
    client_id: Mapped[str] = mapped_column(ForeignKey("clients.client_id"), index=True)
    title: Mapped[str]
    subject: Mapped[str] = mapped_column(index=True)
    grade: Mapped[str] = mapped_column(index=True)
    board: Mapped[str]
    lang: Mapped[str] = mapped_column(index=True)
    source_path: Mapped[str]
    extractor_name: Mapped[str]
    ingested_at: Mapped[datetime] = mapped_column(server_default=func.now())

class Chapter(Base):
    __tablename__ = "chapters"
    chapter_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.book_id", ondelete="CASCADE"), index=True)
    name: Mapped[str]
    order_index: Mapped[int]    # first occurrence position in the book
    # (book_id, name) unique — same chapter name twice in one book is an error
    __table_args__ = (UniqueConstraint("book_id", "name"),)

class Section(Base):
    __tablename__ = "sections"
    section_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.book_id", ondelete="CASCADE"), index=True)
    chapter_id: Mapped[int | None] = mapped_column(ForeignKey("chapters.chapter_id", ondelete="SET NULL"))
    name: Mapped[str]
    order_index: Mapped[int]
    __table_args__ = (UniqueConstraint("book_id", "chapter_id", "name"),)

class ChunkRow(Base):
    __tablename__ = "chunks"
    chunk_id: Mapped[str] = mapped_column(primary_key=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.book_id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(index=True)
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    element_type: Mapped[str]
    chapter_id: Mapped[int | None] = mapped_column(ForeignKey("chapters.chapter_id", ondelete="SET NULL"))
    section_id: Mapped[int | None] = mapped_column(ForeignKey("sections.section_id", ondelete="SET NULL"))
    page_start: Mapped[int]
    page_end: Mapped[int]
    context_snippet: Mapped[str | None]
    image_path: Mapped[str | None]
    lang: Mapped[str] = mapped_column(index=True)