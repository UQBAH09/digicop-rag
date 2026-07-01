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

class ChunkingSettings(BaseSettings):
    """All chunking parameters. Tune via env vars, not code edits."""

    model_config = SettingsConfigDict(
        env_prefix="CHUNKING_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    max_chunk_tokens: int = 500
    chunk_overlap_tokens: int = 75
    context_snippet_chars: int = 200
    tokenizer_model_name: str = "BAAI/bge-m3"

class PostgresSettings(BaseSettings):
    url: str = "postgresql+asyncpg://localhost/digicop"
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env")
