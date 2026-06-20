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


class GlobalBM25:
    """Thin bm25s wrapper that returns doc-ids (not corpus positions)."""

    def __init__(self):
        self._bm25 = None
        self.ids: list[str] = []
        self.id_to_text: dict[str, str] = {}

    def build(self, corpus_ids, corpus_texts):
        import bm25s
        self.ids = list(corpus_ids)
        self.id_to_text = dict(zip(corpus_ids, corpus_texts))
        toks = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
        self._bm25 = bm25s.BM25(k1=1.5, b=0.75)
        self._bm25.index(toks, show_progress=False)

    def topk_ids(self, query_text: str, k: int) -> list:
        if not query_text.split():
            return []
        import bm25s
        qtoks = bm25s.tokenize(query_text, stopwords="en", show_progress=False)
        k = min(k, len(self.ids))
        res, _ = self._bm25.retrieve(qtoks, k=k, show_progress=False)
        return [self.ids[pos] for pos in res[0]]

    def save(self, path: str):
        import json, os
        self._bm25.save(path)
        with open(os.path.join(path, "ids.json"), "w") as f:
            json.dump(self.ids, f)
        with open(os.path.join(path, "id_to_text.json"), "w") as f:
            json.dump(self.id_to_text, f)

    @classmethod
    def load(cls, path: str):
        import bm25s, json, os
        g = cls()
        g._bm25 = bm25s.BM25.load(path)
        with open(os.path.join(path, "ids.json")) as f:
            g.ids = json.load(f)
        with open(os.path.join(path, "id_to_text.json")) as f:
            g.id_to_text = json.load(f)
        return g
