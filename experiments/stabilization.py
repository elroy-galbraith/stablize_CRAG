"""
Stabilization metrics for a single CRAG question.

Given a query and its passage set, we retrieve over every prefix q[1:t] and
derive:
  t_sc  : self-consistency stabilization (smallest t after which top-1 == full-query top-1)
  t_suf : sufficiency stabilization (smallest t whose top-k contains a gold passage)
  phi   : t*/n
  V     : top-1 volatility after the full-query doc is first reached
  H     : hidden latency bound = min(L, max(0, (n - t*) / delta))  (delta in w/s)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from streaming_rag import BM25


@dataclass
class Stab:
    n_words: int
    n_passages: int
    full_top1: int
    t_sc: int
    phi_sc: float
    t_suf: Optional[int]      # None if gold never retrieved (or ungroundable)
    phi_suf: Optional[float]
    volatility: int
    retrieved_gold: bool      # gold appeared in top-k at some prefix


def prefix_records(query: str, passages: list[str], top_k: int = 3,
                   make_retriever=BM25) -> tuple[list[tuple[int, set]], int]:
    """Retrieve over every prefix q[1:t]; return (seq, n) where
    seq[t-1] = (top1_id, set_of_topk_ids). ([], 0) if query or passages empty."""
    words = query.split()
    n = len(words)
    if not passages or n == 0:
        return [], 0
    retriever = make_retriever(passages)
    if hasattr(retriever, "prefix_topk"):
        return retriever.prefix_topk(words, top_k), n
    seq = []
    for t in range(1, n + 1):
        ranked = retriever.topk(" ".join(words[:t]), k=top_k)
        ids = [i for i, _ in ranked]
        seq.append((ids[0] if ids else -1, set(ids)))
    return seq, n


def stabilization(query: str, passages: list[str], gold: set[int], top_k: int = 3,
                  make_retriever=BM25) -> Optional[Stab]:
    """`make_retriever(passages) -> retriever` selects the retriever condition;
    defaults to BM25. A dense retriever (experiments/dense.DenseRetriever) is a
    duck-typed drop-in. All downstream metrics consume only the prefix sequence."""
    seq, n = prefix_records(query, passages, top_k, make_retriever)
    if not seq:
        return None
    full_top1 = seq[-1][0]

    # t_sc: 1 + (last prefix length whose top-1 differs from the full-query top-1)
    last_diff_t = 0
    for idx, (t1, _) in enumerate(seq):
        if t1 != full_top1:
            last_diff_t = idx + 1
    t_sc = last_diff_t + 1

    # volatility: top-1 changes after full_top1 is first reached
    first = next((idx for idx, (t1, _) in enumerate(seq) if t1 == full_top1), None)
    volatility = 0
    if first is not None:
        prev = seq[first][0]
        for idx in range(first + 1, len(seq)):
            if seq[idx][0] != prev:
                volatility += 1
                prev = seq[idx][0]

    # t_suf: first prefix whose top-k intersects the gold passage set
    t_suf = None
    if gold:
        for idx, (_, ids) in enumerate(seq):
            if ids & gold:
                t_suf = idx + 1
                break

    return Stab(
        n_words=n,
        n_passages=len(passages),
        full_top1=full_top1,
        t_sc=t_sc,
        phi_sc=t_sc / n,
        t_suf=t_suf,
        phi_suf=(t_suf / n) if t_suf else None,
        volatility=volatility,
        retrieved_gold=t_suf is not None,
    )


def hidden_latency_ms(t_star: int, n_words: int, L_ms: float, delta_wps: float) -> float:
    """H = min(L, residual_input_time). residual = (n - t*) words / delta wps."""
    residual_ms = max(0.0, (n_words - t_star)) / max(delta_wps, 1e-9) * 1000.0
    return min(L_ms, residual_ms)
