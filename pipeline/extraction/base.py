from abc import ABC, abstractmethod
from pathlib import Path
from shared.models import BookMeta, Document


class UnsupportedLanguageError(ValueError):
    """Raised when a book's language isn't supported by this extractor."""
    pass

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, pdf_path: Path, meta: BookMeta) -> Document: ...