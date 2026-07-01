"""
chunker.py — Core chunking algorithm for the ingestion pipeline.
"""

from __future__ import annotations

import hashlib
import logging
from itertools import groupby
from typing import NamedTuple

from pipeline.chunking.splitter import get_overlap_tail, recursive_split
from shared.config import ChunkingSettings
from shared.models import Chunk, Document, Element
from shared.tokenizer import TokenCounter

logger = logging.getLogger(__name__)


class TaggedElement(NamedTuple):
    element: Element
    chapter: str | None
    section: str | None
    page_no: int
    source_order: int


def walk_elements(document: Document) -> list[TaggedElement]:
    current_chapter: str | None = None
    current_section: str | None = None
    tagged_elements: list[TaggedElement] = []
    source_order: int = 0

    for page in document.pages:
        for element in page.elements:
            if element.type == "heading":
                if element.content.startswith("##"):
                    current_section = element.content[2:].strip()
                elif element.content.startswith("#"):
                    current_chapter = element.content[1:].strip()
                    current_section = None
                else:
                    current_section = element.content.strip()
            else:
                tagged_elements.append(
                    TaggedElement(element, current_chapter, current_section, page.page_no, source_order)
                )
                source_order += 1
    return tagged_elements


def split_streams(
    tagged_elements: list[TaggedElement],
) -> tuple[list[TaggedElement], list[TaggedElement]]:
    text_stream: list[TaggedElement] = []
    standalone_stream: list[TaggedElement] = []
    for tagged in tagged_elements:
        if tagged.element.type == "text":
            text_stream.append(tagged)
        else:
            standalone_stream.append(tagged)
    return text_stream, standalone_stream


def chunk_text_stream(
    text_stream: list[TaggedElement],
    settings: ChunkingSettings,
    counter: TokenCounter,
    lang: str,
    standalone_orders: set[int],
) -> list[dict]:
    chunks: list[dict] = []

    for (chapter, section), group in groupby(
        text_stream, key=lambda t: (t.chapter, t.section)
    ):
        elements = list(group)
        current_content: list[str] = []
        current_tokens: int = 0
        page_start: int = elements[0].page_no
        page_end: int = elements[0].page_no
        source_order: int = elements[0].source_order
        prev_source_order: int = elements[0].source_order

        for tagged in elements:
            if current_content:
                gap_has_standalone = any(
                    prev_source_order < so < tagged.source_order
                    for so in standalone_orders
                )
                if gap_has_standalone:
                    chunks.append({
                        "source_order": source_order,
                        "content": "\n".join(current_content),
                        "page_start": page_start,
                        "page_end": page_end,
                        "chapter": chapter,
                        "section": section,
                        "element_type": "text",
                    })
                    current_content = []
                    current_tokens = 0
                    source_order = tagged.source_order
                    page_start = tagged.page_no
                    page_end = tagged.page_no

            prev_source_order = tagged.source_order
            elem_tokens = counter.count_tokens(text=tagged.element.content)

            if elem_tokens > settings.max_chunk_tokens:
                logger.warning(
                    "Oversized element (%d tokens, max %d) on page %d — recursive split",
                    elem_tokens, settings.max_chunk_tokens, tagged.page_no,
                )
                pieces = recursive_split(
                    tagged.element.content, lang,
                    settings.max_chunk_tokens - settings.chunk_overlap_tokens, counter,
                )
            else:
                pieces = [tagged.element.content]

            for piece in pieces:
                piece_tokens = counter.count_tokens(piece)
                if current_tokens + piece_tokens <= settings.max_chunk_tokens:
                    current_content.append(piece)
                    current_tokens += piece_tokens
                    page_end = tagged.page_no
                else:
                    if current_content:
                        emitted_text = "\n".join(current_content)
                        chunks.append({
                            "source_order": source_order,
                            "content": emitted_text,
                            "page_start": page_start,
                            "page_end": page_end,
                            "chapter": chapter,
                            "section": section,
                            "element_type": "text",
                        })
                        overlap = (
                            get_overlap_tail(emitted_text, settings.chunk_overlap_tokens, counter)
                            if settings.chunk_overlap_tokens > 0
                            else ""
                        )
                    else:
                        overlap = ""

                    if overlap and counter.count_tokens(overlap) + piece_tokens <= settings.max_chunk_tokens:
                        current_content = [overlap, piece]
                        current_tokens = counter.count_tokens(overlap) + piece_tokens
                    else:
                        current_content = [piece]
                        current_tokens = piece_tokens

                    source_order = tagged.source_order
                    page_start = tagged.page_no
                    page_end = tagged.page_no

        if current_content:
            chunks.append({
                "source_order": source_order,
                "content": "\n".join(current_content),
                "page_start": page_start,
                "page_end": page_end,
                "chapter": chapter,
                "section": section,
                "element_type": "text",
            })
    return chunks


def build_standalone_chunks(
    tagged_elements: list[TaggedElement],
    standalone_stream: list[TaggedElement],
    settings: ChunkingSettings,
) -> list[dict]:
    chunks: list[dict] = []
    elem_positions: dict[int, int] = {}
    for i, tagged in enumerate(tagged_elements):
        elem_positions[tagged.source_order] = i

    for tagged in standalone_stream:
        pos = elem_positions[tagged.source_order]
        context_snippet = None
        if pos > 0:
            prev = tagged_elements[pos - 1]
            if prev.element.type in ("text", "equation"):
                context_snippet = prev.element.content[-settings.context_snippet_chars:]

        if tagged.element.type == "figure":
            parts = []
            if tagged.element.content:
                parts.append(tagged.element.content)
            if tagged.element.description:
                parts.append(tagged.element.description)
            content = "\n".join(parts)
        else:
            content = tagged.element.content

        chunks.append({
            "source_order": tagged.source_order,
            "content": content,
            "page_start": tagged.page_no,
            "page_end": tagged.page_no,
            "chapter": tagged.chapter,
            "section": tagged.section,
            "element_type": tagged.element.type,
            "context_snippet": context_snippet,
            "image_path": tagged.element.image_path,
        })
    return chunks


def make_chunk_id(book_id: str, page_start: int, chunk_index: int, content: str) -> str:
    raw = f"{book_id}:{page_start}:{chunk_index}:{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def chunk(
    document: Document, settings: ChunkingSettings, counter: TokenCounter
) -> list[Chunk]:
    tagged_list = walk_elements(document)
    text_stream, standalone_stream = split_streams(tagged_list)
    standalone_orders = {t.source_order for t in standalone_stream}
    text_chunks = chunk_text_stream(
        text_stream, settings, counter, document.meta.lang, standalone_orders
    )
    standalone_chunks = build_standalone_chunks(tagged_list, standalone_stream, settings)

    all_chunk_dicts = text_chunks + standalone_chunks
    all_chunk_dicts.sort(key=lambda c: (c["page_start"], c["source_order"]))

    chunks: list[Chunk] = []
    meta = document.meta
    for index, data in enumerate(all_chunk_dicts):
        chunks.append(Chunk(
            chunk_id=make_chunk_id(meta.book_id, data["page_start"], index, data["content"]),
            content=data["content"],
            element_type=data["element_type"],
            book_id=meta.book_id,
            client_id=meta.client_id,
            subject=meta.subject,
            grade=meta.grade,
            board=meta.board,
            lang=meta.lang,
            chapter=data["chapter"],
            section=data["section"],
            page_start=data["page_start"],
            page_end=data["page_end"],
            chunk_index=index,
            context_snippet=data.get("context_snippet"),
            image_path=data.get("image_path"),
        ))

    type_counts: dict[str, int] = {}
    for c in chunks:
        type_counts[c.element_type] = type_counts.get(c.element_type, 0) + 1
    logger.info(
        "Chunked '%s': %d chunks (%s)",
        document.meta.title, len(chunks),
        ", ".join(f"{t}={n}" for t, n in sorted(type_counts.items())),
    )
    return chunks
