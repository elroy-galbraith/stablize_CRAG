"""Per-word feature/label extraction for the Component A classifier trigger.

Reuses stabilization.prefix_records for the retrieval-side signal and spaCy NER
(lazy import, [trigger] extra) for the entity-side signal. See
docs/superpowers/specs/2026-06-20-component-a-trigger-design.md.
"""
from __future__ import annotations

from typing import Optional

_QWORDS = ("who", "what", "when", "where", "which", "why", "how")


def question_word_type(query: str) -> str:
    for w in query.lower().split()[:3]:
        if w in _QWORDS:
            return w
    return "other"


def char_spans_to_word_offsets(query: str, char_spans: list[tuple[int, int]]) -> list[int]:
    """Map each entity char-start to the 1-based index of the word it lands in."""
    # Build (char_start, word_index) for each whitespace-split word.
    bounds = []
    pos = 0
    for idx, word in enumerate(query.split(), start=1):
        start = query.index(word, pos)
        bounds.append((start, start + len(word), idx))
        pos = start + len(word)
    offsets = set()
    for cstart, _cend in char_spans:
        for wstart, wend, widx in bounds:
            if wstart <= cstart < wend:
                offsets.add(widx)
                break
    return sorted(offsets)


def first_ne_position(offsets: list[int]) -> Optional[int]:
    return min(offsets) if offsets else None


_NLP = None


def _load_nlp():
    """Lazily load and cache the spaCy pipeline ([trigger] extra)."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "spaCy not installed. Install with: uv sync --extra trigger"
            ) from e
        try:
            _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "parser"])
        except OSError as e:
            raise OSError(
                "spaCy model missing. Run: python -m spacy download en_core_web_sm"
            ) from e
    return _NLP


def spacy_ner(query: str) -> list[int]:
    """1-based word offsets of named-entity starts in the full query."""
    if not query.split():
        return []
    doc = _load_nlp()(query)
    spans = [(ent.start_char, ent.start_char) for ent in doc.ents]
    return char_spans_to_word_offsets(query, spans)
