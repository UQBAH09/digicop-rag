from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions ,TableStructureOptions
import pymupdf
import tempfile
import time
import logging
from pathlib import Path

from shared.config import extractor_settings
from pipeline.extraction.base import BaseExtractor, UnsupportedLanguageError
from shared.models import Element, Page, BookMeta, Document
class DoclingExtractor(BaseExtractor):
    LABEL_MAP = {
            "text": "text",
            "list_item": "text",
            "section_header": "heading",
            "table": "table",
            "formula": "equation",
            "picture": "figure",
        }
    
    def __init__(self):
        opts = PdfPipelineOptions()

        opts.do_ocr = True
        opts.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=extractor_settings.docling_force_full_page_ocr)
        opts.images_scale = extractor_settings.docling_images_scale

        opts.do_table_structure = True
        opts.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )

        opts.do_formula_enrichment = extractor_settings.docling_formula_enrichment

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

        self.logger = logging.getLogger(__name__)

    def extract(self, pdf_path: str, meta: BookMeta) -> Document:
        if meta.lang == "ur":
            raise UnsupportedLanguageError("DoclingExtractor does not support Urdu.")

        # Count total pages
        with pymupdf.open(str(pdf_path)) as src:
            total_pages = src.page_count

            pages: list[Page] = []

            for page_idx in range(total_pages):
                page_no = page_idx + 1
                start = time.time()

                tmp_path = None
                try:
                    # Extract single page to a temp PDF
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        single = pymupdf.open()
                        single.insert_pdf(src, from_page=page_idx, to_page=page_idx)
                        single.save(tmp.name)
                        single.close()
                        tmp_path = tmp.name

                    # Convert the single-page PDF
                    result = self.converter.convert(tmp_path)
                    doc_result = result.document

                    # Walk items and build elements
                    elements: list[Element] = []
                    last_noncaption: Element | None = None

                    for item, level in doc_result.iterate_items():
                        if item.label == "caption":
                            if last_noncaption and last_noncaption.type in ("table", "figure"):
                                caption_text = item.text or ""
                                if caption_text and caption_text not in last_noncaption.content:
                                    if last_noncaption.content:
                                        last_noncaption.content = caption_text + "\n\n" + last_noncaption.content
                                    else:
                                        last_noncaption.content = caption_text
                            continue

                        element_type = LABEL_MAP.get(item.label)
                        if element_type is None:
                            continue

                        if element_type == "heading":
                            prefix = "# " if level == 1 else "## "
                            element = Element(type="heading", content=prefix + item.text)
                        elif element_type == "table":
                            md = item.export_to_markdown(doc_result)
                            element = Element(type="table", content=md)
                        elif element_type == "equation":
                            element = Element(type="equation", content=item.text)
                        elif element_type == "figure":
                            element = Element(type="figure", content="")
                        else:
                            element = Element(type="text", content=item.text)

                        last_noncaption = element
                        elements.append(element)

                    char_count = sum(len(e.content) for e in elements)
                    pages.append(Page(
                        page_no=page_no,
                        elements=elements,
                        source="ocr",
                        lang=meta.lang,
                        char_count=char_count
                    ))

                    elapsed = time.time() - start
                    print(f"  [{page_no}/{total_pages}] {len(elements)} elements, {char_count} chars ({elapsed:.1f}s)")

                except Exception as e:
                    # One bad page must never kill the book
                    pages.append(Page(
                        page_no=page_no,
                        elements=[],
                        source="ocr",
                        lang=meta.lang,
                        char_count=0
                    ))
                    self.logger.error(f"{meta.book_id} | page {page_no} FAILED: {e}")
                    print(f"  [{page_no}/{total_pages}] FAILED: {e}")

                finally:
                    if tmp_path:
                        Path(tmp_path).unlink(missing_ok=True)

        self.logger.info(
            f"{meta.book_id} | {len(pages)} pages | "
            f"{sum(len(p.elements) for p in pages)} elements"
        )

        return Document(
            meta=meta,
            pages=pages,
            source_path=str(pdf_path),
            extractor_name="docling_extractor-1.0/tesseract-cli"
        )