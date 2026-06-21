"""Global-corpus tool-intent stabilization on BEIR-NQ (bm25s).

Retrieves query PREFIXES against the whole NQ corpus (~2.68M passages) and against
a per-question pool, to separate real early-stabilization from the candidate-pool
artifact. See docs/superpowers/specs/2026-06-21-global-corpus-tis-design.md.
"""
from __future__ import annotations

from typing import Iterable, Optional

from stabilization import prefix_records


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
        # NOTE: the global arm scores with bm25s (Lucene-style BM25); the per-question
        # arm (perq_t_suf) scores with the vendored pure-Python BM25 via prefix_records.
        # Both use k1=1.5/b=0.75, but the engines differ slightly, so the global-vs-perq
        # gap is corpus-structure-DOMINATED, not purely so. The headline early-vs-late
        # effect (phi_suf ~0.1 per-question vs ~0.75 global) dwarfs any scoring nuance.
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


def global_t_suf(query: str, gold_ids: set, index: "GlobalBM25", k: int) -> Optional[int]:
    words = query.split()
    for t in range(1, len(words) + 1):
        hits = set(index.topk_ids(" ".join(words[:t]), k))
        if hits & gold_ids:
            return t
    return None


def perq_t_suf(query: str, gold_ids: set, index: "GlobalBM25", k: int, n_pool: int) -> Optional[int]:
    # Pool = top-N(full query) U gold, materialized as passage texts.
    pool_ids = list(dict.fromkeys(index.topk_ids(query, n_pool) + list(gold_ids)))
    pool_texts = [index.id_to_text.get(i, "") for i in pool_ids]
    gold_pos = {i for i, did in enumerate(pool_ids) if did in gold_ids}
    seq, n = prefix_records(query, pool_texts, top_k=k)
    if not seq:
        return None
    for t, (_, ids) in enumerate(seq, start=1):
        if ids & gold_pos:
            return t
    return None


def _cell(rows, t_key, phi_key) -> dict:
    sub = [r for r in rows if r[t_key] is not None]
    if not sub:
        return {"n": 0}
    import statistics as st
    phi = [r[phi_key] for r in sub]
    return {"n": len(sub), "phi_suf_mean": round(st.mean(phi), 4),
            "phi_suf_median": round(st.median(phi), 4),
            "t_suf_eq_1_rate": round(sum(1 for r in sub if r[t_key] == 1) / len(sub), 4)}


def summarize(rows: list) -> dict:
    return {"global": _cell(rows, "t_suf_global", "phi_suf_global"),
            "perq": _cell(rows, "t_suf_perq", "phi_suf_perq")}


def main():
    import argparse, csv, json, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-pool", type=int, default=100)
    ap.add_argument("--limit-corpus", type=int, default=None)
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--index-dir", default="results/global/nq_bm25")
    ap.add_argument("--out", default="results/global/nq_tsuf.csv")
    ap.add_argument("--summary-out", default="results/global/nq_tsuf.summary.json")
    args = ap.parse_args()

    corpus_ids, corpus_texts, queries, qrels = load_beir_nq(args.limit_corpus)
    if os.path.isdir(args.index_dir):
        index = GlobalBM25.load(args.index_dir)
    else:
        index = GlobalBM25(); index.build(corpus_ids, corpus_texts)
        os.makedirs(args.index_dir, exist_ok=True); index.save(args.index_dir)

    rows = []
    for qid, gold_ids in qrels.items():
        if qid not in queries:
            continue
        q = queries[qid]
        nwords = max(len(q.split()), 1)
        tg = global_t_suf(q, gold_ids, index, args.k)
        tp = perq_t_suf(q, gold_ids, index, args.k, args.n_pool)
        rows.append({"qid": qid, "n_words": nwords,
                     "t_suf_global": tg, "phi_suf_global": round(tg / nwords, 4) if tg else None,
                     "t_suf_perq": tp, "phi_suf_perq": round(tp / nwords, 4) if tp else None})
        if args.limit_queries and len(rows) >= args.limit_queries:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    s = summarize(rows)
    with open(args.summary_out, "w") as f:
        json.dump({"params": {"k": args.k, "n_pool": args.n_pool,
                              "limit_corpus": args.limit_corpus, "n_queries": len(rows)},
                   "dual": s}, f, indent=2)
    print(f"global: {s['global']}  | perq: {s['perq']}")


if __name__ == "__main__":
    main()
