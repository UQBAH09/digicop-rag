from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions ,TableStructureOptions
import pymupdf
import tempfile
import time
import logging
from pathlib import Path
from PIL import Image
import io

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

                        element_type = self.LABEL_MAP.get(item.label)
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
                            image_path = None
                            if item.prov:
                                try:
                                    prov = item.prov[0]
                                    bbox = prov.bbox
                                    page_obj = src[page_idx]
                                    page_height = page_obj.rect.height

                                    x0, y0 = bbox.l, page_height - bbox.t
                                    x1, y1 = bbox.r, page_height - bbox.b
                                    width, height = abs(x1 - x0), abs(y1 - y0)

                                    if width >= extractor_settings.min_figure_size_px and height >= extractor_settings.min_figure_size_px:
                                        fig_dir = Path(extractor_settings.data_root) / "figures" / meta.book_id
                                        fig_dir.mkdir(parents=True, exist_ok=True)
                                        fig_count = sum(1 for e in elements if e.type == "figure")
                                        fig_filename = f"p{page_no}_{fig_count}.png"

                                        pixmap = page_obj.get_pixmap(dpi=150)
                                        img = Image.open(io.BytesIO(pixmap.tobytes("png")))
                                        scale = 150 / 72
                                        crop = img.crop((x0 * scale, y0 * scale, x1 * scale, y1 * scale))
                                        crop.save(str(fig_dir / fig_filename))
                                        image_path = f"figures/{meta.book_id}/{fig_filename}"
                                    else:
                                        continue  # skip tiny decorative images entirely
                                except Exception as e:
                                    self.logger.warning(f"Could not save figure on page {page_no}: {e}")

                            element = Element(type="figure", content="", image_path=image_path)
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

        document = Document(
            meta=meta,
            pages=pages,
            source_path=str(pdf_path),
            extractor_name="docling_extractor-1.0/tesseract-cli"
        )

        output_path = Path(extractor_settings.data_root) / "extracted" / f"{meta.book_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        self.logger.info(f"Saved to {output_path}")

        return document