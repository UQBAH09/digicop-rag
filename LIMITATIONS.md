# Known Limitations — DoclingExtractor

## Urdu unsupported
Tesseract cannot read Nastaliq script. Urdu books must route to the paid-tier
GeminiOcrExtractor. DoclingExtractor raises UnsupportedLanguageError for lang="ur".

## Speed
- ~5–15 sec/page depending on equation density (formula enrichment is the bottleneck)
- A 300-page book: ~25–75 minutes single-threaded
- Production: per-book Celery parallelism, not per-page

## No bbox / font metadata
Docling abstracts these away in its export. bbox, font_size, is_bold are always None
on OCR pages. The chunker already tolerates this.

## Tesseract OCR artifacts
Scanned pages occasionally produce text fragments ("bg", "of", "a") from decorative
elements or poor scan quality. Not filtered — faithful to what OCR returns.

## Table header garbling
Tesseract sometimes mangles the top row of tables.
**Measured rate:** TODO — measure on full book run and fill in.

## Equation error rate
Formula enrichment produces LaTeX but with errors on scanned inputs
(½→γ₂, √ position errors, misread symbols).
**Measured rate:** TODO — sample 20 equation pages from full book run and fill in.

## Figure detection
- Figures are cropped using Docling's detected bounding box. Accuracy depends on
  layout detection quality.
- Decorative images < 50px (configurable) are filtered out.
- Docling does not extract embedded raster images directly; we crop from rendered pages.