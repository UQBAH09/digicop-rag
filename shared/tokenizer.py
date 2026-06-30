"""
Tokenizer abstraction for the chunking pipeline.

The chunker measures chunk size in tokens, not characters.
This module provides the interface and concrete implementation, keeping the
heavy `transformers` dependency isolated so that:

    1. If the embedding model changes, only the concrete class here needs updating.
    2. Tests pass in a lightweight fake instead of loading a ~2GB model.

CRITICAL: the `transformers` import lives inside BgeM3TokenCounter.__init__,
NOT at module top.
"""

from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    """Anything that can count tokens in a piece of text."""

    def count_tokens(self, text: str) -> int: ...


class BgeM3TokenCounter:
    """Production token counter using the bge-m3 embedding model's tokenizer."""

    def __init__(self) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))
