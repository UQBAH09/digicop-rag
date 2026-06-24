from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractorSettings(BaseSettings):
    docling_images_scale: float = 2.0
    docling_formula_enrichment: bool = True
    docling_force_full_page_ocr: bool = True
    header_footer_repetition_threshold: float = 0.4
    min_figure_size_px: int = 50
    data_root: str = "data"
    model_config = SettingsConfigDict(env_prefix="EXTRACTOR_", env_file=".env")


extractor_settings = ExtractorSettings()