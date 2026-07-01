"""
Generic extractor contract tests.

These validate that ANY BaseExtractor implementation produces output
conforming to the frozen contracts in shared/models.py.

To test a new extractor:
  1. Add a fixture that returns an instance of your extractor
  2. The tests automatically run against it

Usage:
  EXTRACTOR_DOCLING_FORMULA_ENRICHMENT=false uv run pytest tests/pipeline/test_extractor_contract.py -v
"""

import pytest
from pathlib import Path
from shared.models import Document, Page, Element, BookMeta
from pipeline.extraction.base import BaseExtractor, UnsupportedLanguageError


# ---------------------------------------------------------------------------
# Fixtures — add new extractors here
# ---------------------------------------------------------------------------

TEST_PDF = "data/books/test.pdf"  # 3-page fixture, must exist

@pytest.fixture
def test_meta() -> BookMeta:
    return BookMeta(
        book_id="contract-test",
        title="Test Book",
        subject="Physics",
        grade="11",
        board="STBB",
        lang="en",
        client_id="test-user"
    )


@pytest.fixture
def docling_extractor():
    """Skip if Docling is not installed or test PDF missing."""
    if not Path(TEST_PDF).exists():
        pytest.skip(f"Test PDF not found: {TEST_PDF}")
    try:
        from pipeline.extraction.docling_extractor import DoclingExtractor
        return DoclingExtractor()
    except ImportError:
        pytest.skip("DoclingExtractor not available")


# Add future extractors here:
# @pytest.fixture
# def gemini_extractor():
#     from pipeline.extraction.gemini_extractor import GeminiExtractor
#     return GeminiExtractor()


# Parametrize all contract tests to run against every extractor
@pytest.fixture(params=["docling_extractor"])
def extractor(request) -> BaseExtractor:
    return request.getfixturevalue(request.param)


@pytest.fixture
def extracted_doc(extractor, test_meta) -> Document:
    """Run extraction once, reuse across tests in the same session."""
    return extractor.extract(TEST_PDF, test_meta)


# ---------------------------------------------------------------------------
# Contract tests — must pass for ANY extractor
# ---------------------------------------------------------------------------

class TestDocumentStructure:
    """The returned Document must match the frozen contract."""

    def test_returns_document(self, extracted_doc):
        assert isinstance(extracted_doc, Document)

    def test_has_pages(self, extracted_doc):
        assert len(extracted_doc.pages) > 0, "Document must have at least one page"

    def test_meta_preserved(self, extracted_doc, test_meta):
        """Extractor must not alter the BookMeta it was given."""
        assert extracted_doc.meta.book_id == test_meta.book_id
        assert extracted_doc.meta.title == test_meta.title
        assert extracted_doc.meta.subject == test_meta.subject
        assert extracted_doc.meta.lang == test_meta.lang

    def test_extractor_name_set(self, extracted_doc):
        assert extracted_doc.extractor_name, "extractor_name must not be empty"

    def test_source_path_set(self, extracted_doc):
        assert extracted_doc.source_path, "source_path must not be empty"


class TestPageContract:
    """Every Page must conform to the frozen contract."""

    def test_page_numbers_are_1_based(self, extracted_doc):
        for page in extracted_doc.pages:
            assert page.page_no >= 1, f"Page number must be 1-based, got {page.page_no}"

    def test_page_numbers_are_sequential(self, extracted_doc):
        page_nos = [p.page_no for p in extracted_doc.pages]
        assert page_nos == sorted(page_nos), "Pages must be in order"

    def test_no_duplicate_page_numbers(self, extracted_doc):
        page_nos = [p.page_no for p in extracted_doc.pages]
        assert len(page_nos) == len(set(page_nos)), "Duplicate page numbers found"

    def test_source_is_valid(self, extracted_doc):
        for page in extracted_doc.pages:
            assert page.source in ("digital", "ocr"), f"Invalid source: {page.source}"

    def test_lang_matches_meta(self, extracted_doc, test_meta):
        for page in extracted_doc.pages:
            assert page.lang == test_meta.lang

    def test_char_count_matches_elements(self, extracted_doc):
        """char_count must equal sum of len(e.content) for all elements."""
        for page in extracted_doc.pages:
            expected = sum(len(e.content) for e in page.elements)
            assert page.char_count == expected, (
                f"Page {page.page_no}: char_count={page.char_count} "
                f"but sum of element content={expected}"
            )


class TestElementContract:
    """Every Element must conform to the frozen contract."""

    # VALID_TYPES = {"text", "heading", "table", "figure", "equation"}
    VALID_TYPES = {"text", "heading", "table", "equation"}

    def test_element_types_are_valid(self, extracted_doc):
        for page in extracted_doc.pages:
            for e in page.elements:
                assert e.type in self.VALID_TYPES, (
                    f"Page {page.page_no}: invalid element type '{e.type}'"
                )

    def test_headings_have_hash_prefix(self, extracted_doc):
        """Chunker depends on # and ## to detect chapter vs section."""
        for page in extracted_doc.pages:
            for e in page.elements:
                if e.type == "heading":
                    assert e.content.startswith("# ") or e.content.startswith("## "), (
                        f"Page {page.page_no}: heading missing #/## prefix: '{e.content[:50]}'"
                    )

    def test_text_elements_not_empty(self, extracted_doc):
        """Text elements should have content (empty = extractor bug)."""
        empty_count = 0
        total_text = 0
        for page in extracted_doc.pages:
            for e in page.elements:
                if e.type == "text":
                    total_text += 1
                    if not e.content.strip():
                        empty_count += 1
        # Allow up to 5% empty (OCR artifacts) but not more
        if total_text > 0:
            empty_pct = empty_count / total_text
            assert empty_pct < 0.05, (
                f"{empty_count}/{total_text} text elements are empty ({empty_pct:.0%})"
            )

    def test_table_elements_have_markdown(self, extracted_doc):
        """Table content should contain pipe characters (markdown table)."""
        for page in extracted_doc.pages:
            for e in page.elements:
                if e.type == "table" and e.content:
                    assert "|" in e.content, (
                        f"Page {page.page_no}: table has no pipe-markdown"
                    )

    def test_equation_elements_not_empty(self, extracted_doc):
        for page in extracted_doc.pages:
            for e in page.elements:
                if e.type == "equation":
                    assert e.content.strip(), (
                        f"Page {page.page_no}: empty equation element"
                    )

    def test_no_extra_fields(self, extracted_doc):
        """ConfigDict(extra='forbid') should prevent unknown fields."""
        with pytest.raises(Exception):
            Element(type="text", content="test", unknown_field="oops")


class TestUrduRefusal:
    """Every free-tier extractor must refuse Urdu."""

    def test_urdu_raises(self, extractor):
        urdu_meta = BookMeta(
            book_id="urdu-test",
            title="Test",
            subject="Urdu",
            grade="11",
            board="STBB",
            lang="ur",
            client_id='test-user'
        )
        with pytest.raises(UnsupportedLanguageError):
            extractor.extract(TEST_PDF, urdu_meta)

    def test_english_accepted(self, extractor, test_meta):
        """Sanity: English must not raise."""
        doc = extractor.extract(TEST_PDF, test_meta)
        assert isinstance(doc, Document)


class TestHeaderFooterRemoval:
    """Repeated headers/footers should not appear in output."""

    def test_no_element_on_every_page(self, extracted_doc):
        """No single element text should appear on every page (that's a header/footer)."""
        if len(extracted_doc.pages) < 3:
            pytest.skip("Need 3+ pages to test header/footer removal")

        from collections import Counter
        text_counts: Counter[str] = Counter()
        for page in extracted_doc.pages:
            seen = set()
            for e in page.elements:
                normalized = e.content.strip().lower()
                if normalized and normalized not in seen:
                    text_counts[normalized] += 1
                    seen.add(normalized)

        total_pages = len(extracted_doc.pages)
        for text, count in text_counts.items():
            assert count < total_pages, (
                f"'{text[:50]}...' appears on all {total_pages} pages — header/footer not removed"
            )


class TestJsonPersistence:
    """Extraction must produce a valid JSON file."""

    def test_json_file_created(self, extracted_doc, test_meta):
        json_path = Path("data/extracted") / f"{test_meta.book_id}.json"
        assert json_path.exists(), f"JSON file not created: {json_path}"

    def test_json_round_trip(self, extracted_doc, test_meta):
        """JSON must deserialize back to an identical Document."""
        json_path = Path("data/extracted") / f"{test_meta.book_id}.json"
        loaded = Document.model_validate_json(
            json_path.read_text(encoding="utf-8")
        )
        assert len(loaded.pages) == len(extracted_doc.pages)
        assert loaded.meta.book_id == extracted_doc.meta.book_id
        for p1, p2 in zip(loaded.pages, extracted_doc.pages):
            assert p1.page_no == p2.page_no
            assert p1.char_count == p2.char_count
            assert len(p1.elements) == len(p2.elements)