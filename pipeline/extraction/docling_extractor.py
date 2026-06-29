from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    TableStructureOptions,
)

from shared.config import extractor_settings
from pipeline.extraction.base import BaseExtractor, UnsupportedLanguageError
from shared.models import Element, Page, BookMeta, Document
class DoclingExtractor(BaseExtractor):
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

    def extract(self, pdf_path: str, meta: BookMeta) -> Document:
        if meta.lang == "ur":
            raise UnsupportedLanguageError("DoclingExtractor does not support Urdu.")
        
        result = self.converter.convert(str(pdf_path))
        doc = result.document

        LABEL_MAP = {
            "text": "text",
            "list_item": "text",
            "section_header": "heading",
            "table": "table",
            "formula": "equation",
            "picture": "figure",
        }

        pages_dict: dict[int, list[Element]] = {}
        # Track the last table/picture element so we can attach its caption
        last_noncaption_element: Element | None = None

        for item, level in doc.iterate_items():
            page_no = item.prov[0].page_no if item.prov else 1

            # Captions are separate items — attach to the previous table/figure
            if item.label == "caption":
                if last_noncaption_element and last_noncaption_element.type in ("table", "figure"):
                    # Prefix the caption to the existing content
                    caption_text = item.text or ""
                    if last_noncaption_element.content:
                        last_noncaption_element.content = caption_text + "\n\n" + last_noncaption_element.content
                    else:
                        last_noncaption_element.content = caption_text
                continue

            element_type = LABEL_MAP.get(item.label)
            if element_type is None:
                continue

            if element_type == "heading":
                prefix = "# " if level == 1 else "## "
                element = Element(type="heading", content=prefix + item.text)

            elif element_type == "table":
                md = item.export_to_markdown()
                element = Element(type="table", content=md)

            elif element_type == "equation":
                element = Element(type="equation", content=item.text)

            elif element_type == "figure":
                element = Element(type="figure", content="")

            else:
                element = Element(type="text", content=item.text)

            last_noncaption_element = element

            if page_no not in pages_dict:
                pages_dict[page_no] = []
            pages_dict[page_no].append(element)

        pages: list[Page] = []
        for page in sorted(pages_dict.keys()):
            elements = pages_dict[page]
            char_count = sum(len(e.content) for e in elements)
            pages.append(Page(
                page_no=page,
                elements=elements,
                source="ocr",
                lang=meta.lang,
                char_count=char_count
            ))

        return Document(
            meta=meta,
            pages=pages,
            source_path=str(pdf_path),
            extractor_name="docling_extractor-1.0/tesseract-cli"
        )
