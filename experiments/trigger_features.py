"""Per-word feature/label extraction for the Component A classifier trigger.

Reuses stabilization.prefix_records for the retrieval-side signal and spaCy NER
(lazy import, [trigger] extra) for the entity-side signal. See
docs/superpowers/specs/2026-06-20-component-a-trigger-design.md.
"""
from __future__ import annotations

import argparse
import csv
from typing import Optional

from crag import load_crag
from stabilization import prefix_records, stabilization

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


def per_word_rows(meta: dict, seq: list[tuple[int, set]], n: int,
                  t_suf: Optional[int], t_sc: int, ne_offsets: list[int]) -> list[dict]:
    """Build per-word feature/label rows for the Component A trigger.

    Args:
        meta: dict with keys: interaction_id, question_type, domain, query
        seq: list of (top1_id, all_ids) tuples, one per prefix length 1..n
        n: total number of words in the query
        t_suf: word position where top-1 stabilizes on gold (or None if ungroundable)
        t_sc: word position where top-1 self-consistency achieved
        ne_offsets: sorted 1-based word indices where named entities start

    Returns:
        list of dicts, one per word position t=1..n, with keys:
        interaction_id, question_type, domain, retrieved_gold, n_words, t, t_suf, t_sc,
        top1_stable_streak, top1_changed, named_entity_detected, words_since_first_ne,
        question_word_type, label, label_sc
    """
    qword = question_word_type(meta["query"])
    first_ne = first_ne_position(ne_offsets)
    rows = []
    streak = 1
    for i in range(n):
        t = i + 1
        top1 = seq[i][0]
        changed = 1 if (i > 0 and seq[i - 1][0] != top1) else 0
        streak = 1 if changed else (streak + 1 if i > 0 else 1)
        ne_det = 1 if (first_ne is not None and t >= first_ne) else 0
        since = max(0, t - first_ne) if (first_ne is not None and t >= first_ne) else 0
        rows.append({
            "interaction_id": meta["interaction_id"],
            "question_type": meta["question_type"],
            "domain": meta["domain"],
            "retrieved_gold": t_suf is not None,
            "n_words": n,
            "t": t,
            "t_suf": t_suf if t_suf is not None else "",
            "t_sc": t_sc,
            "top1_stable_streak": streak,
            "top1_changed": changed,
            "named_entity_detected": ne_det,
            "words_since_first_ne": since,
            "question_word_type": qword,
            "label": (1 if t >= t_suf else 0) if t_suf is not None else "",
            "label_sc": 1 if t >= t_sc else 0,
        })
    return rows


FEATURE_FIELDS = [
    "interaction_id", "question_type", "domain", "retrieved_gold", "n_words", "t",
    "t_suf", "t_sc", "top1_stable_streak", "top1_changed",
    "named_entity_detected", "words_since_first_ne", "question_word_type",
    "label", "label_sc",
]


def extract_from_examples(examples, top_k: int, ner_fn=spacy_ner) -> list[dict]:
    out = []
    for ex in examples:
        seq, n = prefix_records(ex.query, ex.passages, top_k)
        if not seq:
            continue
        s = stabilization(ex.query, ex.passages, ex.gold, top_k=top_k)
        if s is None:
            continue
        ne_offsets = ner_fn(ex.query)
        meta = {"interaction_id": ex.interaction_id, "question_type": ex.question_type,
                "domain": ex.domain, "query": ex.query}
        out.extend(per_word_rows(meta, seq, n, s.t_suf, s.t_sc, ne_offsets))
    return out


def _examples(benchmark: str, data: Optional[str], split: int, limit: Optional[int]):
    if benchmark == "crag":
        return load_crag(data, split=split, limit=limit)
    from benchmarks import BENCHMARKS
    return BENCHMARKS[benchmark](split, limit=limit)


def extract(data: str, split: int, top_k: int, ner_fn=spacy_ner,
            limit: Optional[int] = None, benchmark: str = "crag") -> list[dict]:
    return extract_from_examples(_examples(benchmark, data, split, limit), top_k, ner_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark", default="crag", choices=["crag", "hotpotqa"])
    args = ap.parse_args()
    rows = extract(args.data, args.split, args.top_k, limit=args.limit, benchmark=args.benchmark)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} per-word rows -> {args.out}")


if __name__ == "__main__":
    main()
