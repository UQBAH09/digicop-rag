import pytest
from shared.models import BookMeta, Document, Element
from pipeline.extraction.base import UnsupportedLanguageError


def test_urdu_raises():
    """Urdu input must raise without invoking Docling."""
    from pipeline.extraction.docling_extractor import DoclingExtractor
    
    meta = BookMeta(
        book_id="test-urdu", title="Test", subject="Urdu",
        grade="11", board="STBB", lang="ur"
    )
    extractor = DoclingExtractor()
    with pytest.raises(UnsupportedLanguageError):
        extractor.extract("data/books/test.pdf", meta)


def test_element_types():
    """Element type must be one of the allowed values."""
    # Valid
    Element(type="text", content="hello")
    Element(type="heading", content="# Title")
    Element(type="table", content="| a | b |")
    Element(type="figure", content="")
    Element(type="equation", content="E = mc^2")

    # Invalid
    with pytest.raises(Exception):
        Element(type="paragraph", content="wrong")


def test_extra_fields_rejected():
    """ConfigDict(extra='forbid') must catch typos."""
    with pytest.raises(Exception):
        Element(tyep="text", content="hello")