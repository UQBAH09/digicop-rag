"""Tests for the chunking pipeline."""

from __future__ import annotations

import pytest

from pipeline.chunking.chunker import chunk, walk_elements, split_streams
from pipeline.chunking.splitter import split_sentences, recursive_split
from shared.config import ChunkingSettings
from shared.models import BookMeta, Chunk, Document, Element, Page
from shared.tokenizer import TokenCounter


class FakeCounter:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def counter() -> FakeCounter:
    return FakeCounter()


@pytest.fixture
def settings() -> ChunkingSettings:
    return ChunkingSettings(max_chunk_tokens=20, chunk_overlap_tokens=5, context_snippet_chars=50)


@pytest.fixture
def meta() -> BookMeta:
    return BookMeta(book_id="test-sci-8", title="Science Grade 8", subject="Science", grade="8", board="FBISE", lang="en")


@pytest.fixture
def multi_section_doc(meta: BookMeta) -> Document:
    return Document(
        meta=meta,
        pages=[
            Page(page_no=1, source="ocr", lang="en", char_count=200, elements=[
                Element(type="text", content="This is the introduction."),
                Element(type="heading", content="# Forces"),
                Element(type="text", content="A force is a push or pull."),
                Element(type="heading", content="## Gravity"),
                Element(type="text", content="Gravity pulls objects toward Earth. It is a fundamental force of nature."),
            ]),
            Page(page_no=2, source="ocr", lang="en", char_count=300, elements=[
                Element(type="text", content="The gravitational constant is important."),
                Element(type="equation", content="F = mg"),
                Element(type="figure", content="Gravity diagram", description="An apple falling from a tree toward the ground.", image_path="/figures/gravity.png"),
                Element(type="table", content="| Planet | g (m/s2) |\n|---|---|\n| Earth | 9.8 |"),
                Element(type="heading", content="## Friction"),
                Element(type="text", content="Friction opposes motion between surfaces."),
            ]),
        ],
        source_path="test.pdf",
        extractor_name="test",
    )


class TestMultiSectionChunking:
    def test_produces_multiple_chunks(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        assert len(chunks) > 1

    def test_introduction_has_no_chapter(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        intro = [c for c in chunks if c.chapter is None and c.section is None]
        assert len(intro) >= 1
        assert "introduction" in intro[0].content.lower()

    def test_section_tags_are_correct(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        assert any(c.section == "Gravity" for c in chunks)
        assert any(c.section == "Friction" for c in chunks)

    def test_chapter_is_set(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        assert any(c.chapter == "Forces" for c in chunks)

    def test_no_chunk_spans_sections(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        for c in chunks:
            if c.section == "Friction":
                assert "gravity" not in c.content.lower() or c.element_type != "text"

    def test_chunk_index_is_sequential(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_book_metadata_on_every_chunk(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        for c in chunks:
            assert c.book_id == "test-sci-8"
            assert c.subject == "Science"
            assert c.grade == "8"
            assert c.board == "FBISE"
            assert c.lang == "en"


class TestOversizedSplitting:
    def test_all_chunks_under_token_cap(self, counter, settings):
        long_text = " ".join(f"word{i}" for i in range(50))
        doc = Document(
            meta=BookMeta(book_id="t", title="T", subject="S", grade="1", board="B", lang="en"),
            pages=[Page(page_no=1, source="ocr", lang="en", char_count=500, elements=[Element(type="text", content=long_text)])],
            source_path="t.pdf", extractor_name="test",
        )
        chunks = chunk(doc, settings, counter)
        for c in chunks:
            assert counter.count_tokens(c.content) <= settings.max_chunk_tokens

    def test_no_mid_word_cuts(self, counter):
        text = "Supercalifragilistic is a very long word in a sentence. Another sentence here."
        pieces = recursive_split(text, "en", 5, counter)
        for piece in pieces:
            assert not piece.startswith(" ")
            assert not piece.endswith(" ")


class TestFigureChunks:
    def test_figure_is_standalone_chunk(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        assert len([c for c in chunks if c.element_type == "figure"]) == 1

    def test_figure_has_image_path(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        fig = [c for c in chunks if c.element_type == "figure"][0]
        assert fig.image_path == "/figures/gravity.png"

    def test_figure_content_has_description(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        fig = [c for c in chunks if c.element_type == "figure"][0]
        assert "Gravity diagram" in fig.content
        assert "apple falling" in fig.content

    def test_figure_has_context_snippet(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        fig = [c for c in chunks if c.element_type == "figure"][0]
        assert fig.context_snippet is not None

    def test_table_is_standalone_chunk(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        tables = [c for c in chunks if c.element_type == "table"]
        assert len(tables) == 1
        assert "Planet" in tables[0].content


class TestUrduSplitting:
    def test_sentence_split_on_urdu_full_stop(self):
        text = "یہ پہلا جملہ ہے۔ یہ دوسرا جملہ ہے۔ یہ تیسرا جملہ ہے۔"
        assert len(split_sentences(text, "ur")) == 3

    def test_urdu_chunking_produces_multiple_chunks(self, counter):
        doc = Document(
            meta=BookMeta(book_id="ur-5", title="Urdu 5", subject="Urdu", grade="5", board="FBISE", lang="ur"),
            pages=[Page(page_no=1, source="ocr", lang="ur", char_count=200, elements=[
                Element(type="heading", content="# پہلا باب"),
                Element(type="text", content="یہ پہلا جملہ ہے۔ یہ دوسرا جملہ ہے۔ یہ تیسرا جملہ ہے۔ یہ چوتھا جملہ ہے۔ یہ پانچواں جملہ ہے۔ یہ چھٹا جملہ ہے۔"),
            ])],
            source_path="urdu.pdf", extractor_name="test",
        )
        settings = ChunkingSettings(max_chunk_tokens=15, chunk_overlap_tokens=3, context_snippet_chars=50)
        assert len(chunk(doc, settings, counter)) > 1

    def test_urdu_chapter_heading_stripped(self, counter):
        doc = Document(
            meta=BookMeta(book_id="ur-5", title="Urdu 5", subject="Urdu", grade="5", board="FBISE", lang="ur"),
            pages=[Page(page_no=1, source="ocr", lang="ur", char_count=100, elements=[
                Element(type="heading", content="# پہلا باب"),
                Element(type="text", content="کچھ متن۔"),
            ])],
            source_path="urdu.pdf", extractor_name="test",
        )
        settings = ChunkingSettings(max_chunk_tokens=50, chunk_overlap_tokens=0)
        assert chunk(doc, settings, counter)[0].chapter == "پہلا باب"


class TestDeterminism:
    def test_same_input_same_ids(self, multi_section_doc, settings, counter):
        ids1 = [c.chunk_id for c in chunk(multi_section_doc, settings, counter)]
        ids2 = [c.chunk_id for c in chunk(multi_section_doc, settings, counter)]
        assert ids1 == ids2

    def test_same_input_same_content(self, multi_section_doc, settings, counter):
        r1 = chunk(multi_section_doc, settings, counter)
        r2 = chunk(multi_section_doc, settings, counter)
        for c1, c2 in zip(r1, r2):
            assert c1.content == c2.content
            assert c1.chapter == c2.chapter
            assert c1.chunk_index == c2.chunk_index

    def test_different_content_different_ids(self, counter):
        def make_doc(text):
            return Document(
                meta=BookMeta(book_id="t", title="T", subject="S", grade="1", board="B", lang="en"),
                pages=[Page(page_no=1, source="ocr", lang="en", char_count=100, elements=[Element(type="text", content=text)])],
                source_path="t.pdf", extractor_name="test",
            )
        settings = ChunkingSettings(max_chunk_tokens=50, chunk_overlap_tokens=0)
        assert chunk(make_doc("Hello"), settings, counter)[0].chunk_id != chunk(make_doc("Goodbye"), settings, counter)[0].chunk_id


class TestPageSpanning:
    def test_chunk_spanning_page_break(self, counter):
        doc = Document(
            meta=BookMeta(book_id="t", title="T", subject="S", grade="1", board="B", lang="en"),
            pages=[
                Page(page_no=1, source="ocr", lang="en", char_count=50, elements=[
                    Element(type="heading", content="# Chapter"),
                    Element(type="text", content="Start of paragraph."),
                ]),
                Page(page_no=2, source="ocr", lang="en", char_count=50, elements=[
                    Element(type="text", content="Continuation on page two."),
                ]),
            ],
            source_path="t.pdf", extractor_name="test",
        )
        settings = ChunkingSettings(max_chunk_tokens=50, chunk_overlap_tokens=0, context_snippet_chars=50)
        spanning = [c for c in chunk(doc, settings, counter) if c.page_start != c.page_end]
        assert len(spanning) >= 1
        assert spanning[0].page_start == 1
        assert spanning[0].page_end == 2


class TestEdgeCases:
    def test_empty_figure_content_still_produces_chunk(self, counter):
        doc = Document(
            meta=BookMeta(book_id="t", title="T", subject="S", grade="1", board="B", lang="en"),
            pages=[Page(page_no=1, source="ocr", lang="en", char_count=50, elements=[
                Element(type="figure", content="", description=None, image_path="/figures/unknown.png"),
            ])],
            source_path="t.pdf", extractor_name="test",
        )
        settings = ChunkingSettings(max_chunk_tokens=50, chunk_overlap_tokens=0)
        figs = [c for c in chunk(doc, settings, counter) if c.element_type == "figure"]
        assert len(figs) == 1
        assert figs[0].image_path == "/figures/unknown.png"

    def test_heading_without_hash_defaults_to_section(self, counter):
        doc = Document(
            meta=BookMeta(book_id="t", title="T", subject="S", grade="1", board="B", lang="en"),
            pages=[Page(page_no=1, source="ocr", lang="en", char_count=100, elements=[
                Element(type="heading", content="# Real Chapter"),
                Element(type="heading", content="Just a Heading"),
                Element(type="text", content="Some text."),
            ])],
            source_path="t.pdf", extractor_name="test",
        )
        settings = ChunkingSettings(max_chunk_tokens=50, chunk_overlap_tokens=0)
        chunks = chunk(doc, settings, counter)
        assert chunks[0].chapter == "Real Chapter"
        assert chunks[0].section == "Just a Heading"


class TestSupervisorRequested:
    def test_consecutive_urdu_delimiters(self):
        assert split_sentences("A۔۔ B؟ C", "ur") == ["A۔۔", "B؟", "C"]

    def test_unpunctuated_tail_preserved(self):
        assert split_sentences("First sentence. Trailing text", "en") == ["First sentence.", "Trailing text"]

    def test_overlap_survives_and_under_budget(self, counter):
        text = " ".join(f"word{i}" for i in range(35))
        doc = Document(
            meta=BookMeta(book_id="t", title="T", subject="S", grade="1", board="B", lang="en"),
            pages=[Page(page_no=1, source="ocr", lang="en", char_count=500, elements=[Element(type="text", content=text)])],
            source_path="t.pdf", extractor_name="test",
        )
        settings = ChunkingSettings(max_chunk_tokens=20, chunk_overlap_tokens=5, context_snippet_chars=50)
        chunks = chunk(doc, settings, counter)
        for c in chunks:
            assert counter.count_tokens(c.content) <= settings.max_chunk_tokens
        assert len(chunks) >= 2
        tail = chunks[0].content.split()[-3:]
        head = chunks[1].content.split()[:5]
        assert any(w in head for w in tail)

    def test_equation_is_standalone_chunk(self, multi_section_doc, settings, counter):
        chunks = chunk(multi_section_doc, settings, counter)
        eqs = [c for c in chunks if c.element_type == "equation"]
        assert len(eqs) == 1
        assert eqs[0].content == "F = mg"
        assert eqs[0].context_snippet is not None


class TestGapDetection:
    def test_table_between_text_elements_causes_split(self, counter):
        doc = Document(
            meta=BookMeta(book_id="t", title="T", subject="S", grade="1", board="B", lang="en"),
            pages=[Page(page_no=1, source="ocr", lang="en", char_count=200, elements=[
                Element(type="heading", content="## Results"),
                Element(type="text", content="The data shows the following."),
                Element(type="table", content="| A | B |\n|---|---|\n| 1 | 2 |"),
                Element(type="text", content="As seen above the values differ."),
            ])],
            source_path="t.pdf", extractor_name="test",
        )
        settings = ChunkingSettings(max_chunk_tokens=50, chunk_overlap_tokens=0, context_snippet_chars=50)
        text_chunks = [c for c in chunk(doc, settings, counter) if c.element_type == "text"]
        assert len(text_chunks) == 2
        assert "following" in text_chunks[0].content
        assert "above" in text_chunks[1].content
