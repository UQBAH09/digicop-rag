# Known Limitations — DoclingExtractor v1.0

## Urdu Unsupported
Tesseract cannot read Nastaliq script. Urdu books (lang="ur") are explicitly
refused with UnsupportedLanguageError. Urdu must route to the paid-tier
GeminiOcrExtractor.

## Speed
- Local laptop (CPU only): ~5-15 sec/page without equations, up to 15 min/page
  with heavy equations (formula enrichment model runs on CPU)
- Google Colab T4 GPU: 101 pages in 28 min (~17 sec/page average)
- Full 325-page book estimate: ~90 min on T4 GPU, ~25+ hours on CPU-only laptop
- Production target: GPU server + per-book Celery parallelism
- Formula enrichment is the bottleneck — pages without equations take 3-8 sec
  regardless of hardware

## No bbox / Font Metadata
Docling abstracts layout coordinates away in its export. bbox, font_size, and
is_bold are always None on OCR pages. The chunker already tolerates this — it
uses heading # markers for structure, not font properties.

## Tesseract OCR Artifacts
Scanned pages occasionally produce single-character or junk text fragments
("bg", "of", "a", "movement") from decorative elements, watermarks, or poor
scan quality. These are not filtered — the extractor stays faithful to what
OCR returns. The chunker's minimum-size thresholds handle most of these.

## Table Header Garbling
Tesseract sometimes mangles the top row of tables, especially when the header
row uses bold/colored text against a background.
**Measured on 101-page run:** 10 tables detected. Minor garbling observed in
column headers (misread characters, merged cells). Not corrected in code —
documented as known limitation.

## Equation Error Rate
Formula enrichment (CodeFormulaV2 model) produces LaTeX but with errors on
scanned inputs. Common issues:
- ½ → γ₂ (fraction misread as gamma)
- √ position errors
- Subscript/superscript confusion
- Misread symbols in handwritten-style equations
**Measured on 101-page run:** 251 equations extracted. With formula enrichment
off, equations produce empty content and are skipped. With enrichment on, LaTeX
is structurally correct on ~80% of equations; remaining ~20% have symbol-level
errors that require manual review.

## Figure Extraction Disabled (v1)
Figure cropping is implemented but disabled for v1 to reduce complexity.
The code detects figures via Docling's layout model, crops using bbox coordinates
from PyMuPDF page renders, and filters decorative images below min_figure_size_px
(default 50px). To enable, uncomment "picture": "figure" in LABEL_MAP.
Caption attachment to preceding figure elements is ready.

## Empty Pages
Some pages (unit dividers, full-page images without OCR text) produce 0 elements
and 0 chars. These are preserved as empty Page objects to maintain page numbering
alignment with the physical book.

## Running Header Detection
The header/footer removal filter uses a repetition threshold (>40% of pages,
minimum 2 pages). On very short documents (<5 pages), detection accuracy drops.
The filter logs all dropped text for audit. Over-aggressive removal has not been
observed on the 101-page test run.

## Docling Version Dependency
Tested with Docling's Tesseract CLI backend. The `export_to_markdown(doc)` API
requires the document reference — omitting it triggers a deprecation warning.
PictureItem.image returns None; figure crops use PyMuPDF rendering instead.