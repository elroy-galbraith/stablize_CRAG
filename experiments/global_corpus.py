"""Global-corpus tool-intent stabilization on BEIR-NQ (bm25s).

Retrieves query PREFIXES against the whole NQ corpus (~2.68M passages) and against
a per-question pool, to separate real early-stabilization from the candidate-pool
artifact. See docs/superpowers/specs/2026-06-21-global-corpus-tis-design.md.
"""
from __future__ import annotations

from typing import Iterable, Optional


def corpus_row_to_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title} {text}".strip()


def qrels_to_dict(rows: Iterable[dict]) -> dict:
    out: dict[str, set] = {}
    for r in rows:
        if int(r["score"]) > 0:
            out.setdefault(r["query-id"], set()).add(r["corpus-id"])
    return out


def load_beir_nq(limit_corpus: Optional[int] = None):
    from datasets import load_dataset  # lazy: [bench] extra

    corpus_ids, corpus_texts = [], []
    cds = load_dataset("BeIR/nq", "corpus", split="corpus", streaming=True)
    for n, row in enumerate(cds):
        if limit_corpus and n >= limit_corpus:
            break
        corpus_ids.append(row["_id"])
        corpus_texts.append(corpus_row_to_text(row))
    queries = {row["_id"]: row["text"]
               for row in load_dataset("BeIR/nq", "queries", split="queries", streaming=True)}
    qrels = qrels_to_dict(load_dataset("BeIR/nq-qrels", split="test", streaming=True))
    return corpus_ids, corpus_texts, queries, qrels
