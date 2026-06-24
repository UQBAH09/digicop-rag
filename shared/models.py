"""
Data models for the DigiCop textbook ingestion pipeline.

Two groups:
    INPUT CONTRACT (frozen — do not modify):
        Element, Page, BookMeta, Document
        These come from the extraction stage and are consumed as-is.

    OUTPUT CONTRACT (frozen after supervisor sign-off):
        Chunk
        Everything downstream (embedding, Qdrant payload, retrieval filters)
        is pinned to this shape.

All models use ConfigDict(extra="forbid") so pydantic rejects unexpected fields
instead of silently swallowing typos.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


# ===========================================================================
# INPUT CONTRACT — produced by the extractor, consumed by the chunker
# ===========================================================================


class Element(BaseModel):
    """A single content element on a page, in reading order.

    The `type` field determines which optional fields are meaningful:
        text/heading  → content holds the prose
        table         → content holds a markdown rendering; table_data holds the grid
        figure        → content holds the caption; image_path + description hold the visual
        equation      → content holds LaTeX
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "heading", "table", "figure", "equation"]
    content: str = ""

    # --- positional / layout ---
    # Available on digital pages and coordinate-aware OCR (e.g. PaddleOCR-VL).
    # None on most scanned/OCR pages — the chunker must NOT depend on these.
    bbox: tuple[float, float, float, float] | None = None
    font_size: float | None = None  # heading-level hint (digital pages only)
    is_bold: bool | None = None  # heading-level hint (digital pages only)

    # --- table-specific ---
    # Structured cell grid kept alongside the markdown in `content`.
    # The markdown is what gets embedded and read by the LLM;
    # table_data preserves cell structure for future use (re-rendering, lookup).
    table_data: list[list[str]] | None = None

    # --- figure-specific ---
    image_path: str | None = None  # path to the cropped figure image
    # Natural-language description written by a vision model.
    # THIS is what gets embedded (you can't embed pixels into a text vector space);
    # the image at image_path is what gets handed to the LLM at answer time.
    description: str | None = None

    # --- provenance ---
    confidence: float | None = None  # OCR confidence where the engine reports it


class Page(BaseModel):
    """One page of a textbook, containing elements in reading order.

    Headers and footers are already removed by the extractor.
    """

    model_config = ConfigDict(extra="forbid")

    page_no: int  # 1-based, matches the printed page number
    elements: list[Element]
    source: Literal["digital", "ocr"]
    lang: str
    char_count: int  # sum of len(e.content) across elements


class BookMeta(BaseModel):
    """Identity metadata for a textbook — shared across all its chunks."""

    model_config = ConfigDict(extra="forbid")

    book_id: str
    title: str
    subject: str
    grade: str
    board: str
    lang: str


class Document(BaseModel):
    """A complete textbook: metadata + ordered pages.

    This is the chunker's input — one Document in, list[Chunk] out.
    """

    model_config = ConfigDict(extra="forbid")

    meta: BookMeta
    pages: list[Page]
    source_path: str
    extractor_name: str


# ===========================================================================
# OUTPUT CONTRACT — produced by the chunker, consumed by the embedder
# ===========================================================================


class Chunk(BaseModel):
    """A single retrieval unit — the atom that gets embedded and stored in Qdrant.

    Design decisions baked into this model:

    1. Metadata is FLATTENED (book_id, subject, grade, etc. on every chunk)
       so Qdrant payload filters can match directly ("grade == 8 AND subject == Science")
       without nested field access.

    2. chunk_id is DETERMINISTIC: hash(book_id, page_start, chunk_index, content).
       Re-ingesting the same book with the same config produces identical IDs,
       so Qdrant upserts are clean no-ops — not silent duplicates.

    3. context_snippet and image_path support the "standalone-with-context" pattern
       for tables and figures. They're None for text chunks.
    """

    model_config = ConfigDict(extra="forbid")

    # --- identity ---
    chunk_id: str
    chunk_index: int  # 0-based order within the book

    # --- the retrievable payload (this is what gets embedded) ---
    content: str
    element_type: Literal["text", "table", "figure", "equation"]

    # --- book identity (flattened for Qdrant payload filters) ---
    book_id: str
    subject: str
    grade: str
    board: str
    lang: str

    # --- structural location ---
    chapter: str | None = None
    section: str | None = None
    page_start: int
    page_end: int  # == page_start unless the chunk spans a page break

    # --- standalone-with-context support (tables/figures only) ---
    context_snippet: str | None = None  # lead-in text from the preceding element
    image_path: str | None = None  # figure chunks: image to hand the LLM at answer time
