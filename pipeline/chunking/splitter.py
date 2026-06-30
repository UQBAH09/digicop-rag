"""
Script-aware text splitting utilities for the chunking pipeline.
"""

from __future__ import annotations

import re

from shared.tokenizer import TokenCounter

SENTENCE_ENDERS: dict[str, str] = {
    "en": ".!?",
    "ur": "۔؟!",
}
_FALLBACK_ENDERS = ".!?۔؟"


def split_sentences(text: str, lang: str) -> list[str]:
    """Split text into sentences, keeping punctuation attached."""
    enders = SENTENCE_ENDERS.get(lang, _FALLBACK_ENDERS)
    escaped = re.escape(enders)
    pattern = rf"[^{escaped}]+[{escaped}]+|[^{escaped}]+$"
    return [m.group().strip() for m in re.finditer(pattern, text) if m.group().strip()]


def recursive_split(
    text: str, lang: str, max_tokens: int, counter: TokenCounter,
) -> list[str]:
    """Break text into pieces that each fit under max_tokens."""
    if counter.count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        result: list[str] = []
        for para in paragraphs:
            result.extend(recursive_split(para, lang, max_tokens, counter))
        return result

    sentences = split_sentences(text, lang)
    if len(sentences) > 1:
        result = []
        for sent in sentences:
            result.extend(recursive_split(sent, lang, max_tokens, counter))
        return result

    words = text.split()
    result = []
    current_words: list[str] = []
    for word in words:
        candidate = " ".join(current_words + [word])
        if counter.count_tokens(candidate) > max_tokens and current_words:
            result.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)
    if current_words:
        result.append(" ".join(current_words))
    return result


def get_overlap_tail(text: str, max_tokens: int, counter: TokenCounter) -> str:
    """Extract the last ~max_tokens worth of text, breaking on word boundaries."""
    words = text.split()
    tail_words: list[str] = []
    for word in reversed(words):
        candidate = [word] + tail_words
        if counter.count_tokens(" ".join(candidate)) > max_tokens:
            break
        tail_words.insert(0, word)
    return " ".join(tail_words)
