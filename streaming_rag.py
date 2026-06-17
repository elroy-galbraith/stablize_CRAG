"""
Streaming RAG (text-mode) — a faithful, hardware-light reproduction of the core
orchestration from "Stream RAG: Instant and Accurate Spoken Dialogue Systems with
Streaming Tool Usage" (Arora et al., arXiv:2510.02044).

What this reproduces (the modality-agnostic contribution):
  - Trigger:   WHEN to fire a tool query, while input is still arriving.
  - Threads:   N speculative tool queries in flight in parallel.
  - Reflector: are the retrieved results sufficient to answer yet?

What this deliberately does NOT do:
  - The end-to-end speech LM or its post-training (out of scope for CPU).
  - Audio I/O (that is the optional next layer: whisper.cpp in, piper out).

Everything here is async, dependency-light (numpy optional, not required), and
designed so the TOOL LAYER is swappable: in-process retrieval vs a simulated MCP
transport, so you can measure MCP's latency impact empirically.

VENDORED COPY — source of truth is the `streamRAG` project. This file is copied
in so the CRAG stabilization study is self-contained and CPU-reproducible. Keep
the paper-derived latency constants in `Config` (query_gen_ms, fuse_ms from
Arora et al. Table 3) in sync if you re-vendor; they drive the RQ3 validation.
"""
from __future__ import annotations

import asyncio
import math
import time
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# Minimal BM25 (pure-python, CPU-friendly, no deps) — stands in for a real tool #
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.toks = [_tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.toks) / max(self.N, 1)
        self.df: dict[str, int] = {}
        for t in self.toks:
            for w in set(t):
                self.df[w] = self.df.get(w, 0) + 1
        self.idf = {
            w: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for w, df in self.df.items()
        }

    def score(self, query: str) -> list[float]:
        q = _tok(query)
        out = []
        for toks in self.toks:
            if not toks:
                out.append(0.0)
                continue
            tf: dict[str, int] = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            dl = len(toks)
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                idf = self.idf.get(w, 0.0)
                num = tf[w] * (self.k1 + 1)
                den = tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * num / den
            out.append(s)
        return out

    def topk(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        scored = list(enumerate(self.score(query)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------- #
# Tool layer — swappable. Direct (in-process) vs simulated MCP transport.       #
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    query: str
    hits: list[tuple[int, str, float]]  # (doc_id, text, score)
    latency_ms: float
    fired_at: float
    returned_at: float

    @property
    def top_score(self) -> float:
        return self.hits[0][2] if self.hits else 0.0


class ToolBroker:
    """Abstract tool. Real deployments back this with web search, a KG API, etc."""

    async def search(self, query: str) -> ToolResult:  # pragma: no cover
        raise NotImplementedError


class DirectRetrievalBroker(ToolBroker):
    """In-process BM25. `exec_latency_ms` simulates a real tool's execution time.
    `transport_overhead_ms` simulates protocol/transport cost (0 for in-process)."""

    def __init__(
        self,
        corpus: list[str],
        exec_latency_ms: float = 400.0,
        transport_overhead_ms: float = 0.0,
        top_k: int = 3,
    ):
        self.corpus = corpus
        self.bm25 = BM25(corpus)
        self.exec_latency_ms = exec_latency_ms
        self.transport_overhead_ms = transport_overhead_ms
        self.top_k = top_k

    @property
    def scorer(self) -> BM25:
        """Re-rank hook: a per-doc scorer over the corpus (higher = better).
        Lets the streaming pipeline re-rank speculative results against the
        full query. Vector brokers expose the same `.scorer` interface."""
        return self.bm25

    async def search(self, query: str) -> ToolResult:
        fired = time.perf_counter()
        # transport hop happens first (think: client -> MCP server), then exec.
        await asyncio.sleep(
            (self.transport_overhead_ms + self.exec_latency_ms) / 1000.0
        )
        hits = [
            (i, self.corpus[i], sc) for i, sc in self.bm25.topk(query, self.top_k)
        ]
        returned = time.perf_counter()
        return ToolResult(
            query=query,
            hits=hits,
            latency_ms=(returned - fired) * 1000.0,
            fired_at=fired,
            returned_at=returned,
        )


class MCPSimBroker(DirectRetrievalBroker):
    """Same retrieval, but models an MCP hop. Local stdio ~ a few ms;
    remote SSE/HTTP ~ tens to hundreds of ms. Set transport_overhead_ms to taste."""

    def __init__(self, corpus, transport_overhead_ms=80.0, **kw):
        super().__init__(corpus, transport_overhead_ms=transport_overhead_ms, **kw)


# --------------------------------------------------------------------------- #
# Reflector — "are these results sufficient to answer the (current) query?"     #
# --------------------------------------------------------------------------- #
class Reflector:
    """Heuristic by default; swap `.sufficient` for an LLM call in production."""

    def __init__(self, score_threshold: float = 2.0, coverage_threshold: float = 0.4):
        self.score_threshold = score_threshold
        self.coverage_threshold = coverage_threshold

    def sufficient(
        self, result: Optional[ToolResult], query: str, scorer: Optional["BM25"] = None
    ) -> bool:
        if result is None or not result.hits:
            return False
        # Heuristic: top hit clears a relevance bar AND covers the query's
        # content words. A production reflector would prompt an LLM here.
        if _relevance(result, query, scorer) < self.score_threshold:
            return False
        q_terms = set(_tok(query))
        doc_terms = set(_tok(result.hits[0][1]))
        coverage = len(q_terms & doc_terms) / max(len(q_terms), 1)
        return coverage >= self.coverage_threshold


# --------------------------------------------------------------------------- #
# Config + metrics                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # §F: speech processed at ~150 wpm ≈ 2.5 words/s
    words_per_sec: float = 2.5
    # §3.2.1: Fixed-Interval fires every 1 s; at 2.5 wps that is ~2-3 words
    trigger_interval_words: int = 2
    # Figure 4: Fixed-Interval averages 4.2 parallel threads
    max_threads: int = 4
    # Table 3 P50: LLM time to generate the tool query from partial audio
    query_gen_ms: float = 590.0
    # Table 3 P50: LLM time to generate the spoken response from tool results
    fuse_ms: float = 2520.0


@dataclass
class Metrics:
    mode: str
    perceived_ms: float               # time from END OF INPUT to answer-ready
    wall_ms: float                    # total including input streaming
    calls_fired: int
    calls_used: int
    answer_doc: int
    transport_overhead_ms: float

    @property
    def calls_wasted(self) -> int:
        return max(self.calls_fired - self.calls_used, 0)


# --------------------------------------------------------------------------- #
# Input stream simulator                                                        #
# --------------------------------------------------------------------------- #
async def _stream_words(text: str, wps: float):
    partial: list[str] = []
    for w in text.split():
        partial.append(w)
        await asyncio.sleep(1.0 / wps)
        yield " ".join(partial)


def _relevance(result: Optional[ToolResult], query: str, scorer: Optional[BM25]) -> float:
    """Relevance of a result's top doc to the (full) query, scored against the
    real corpus. Falls back to the result's own top score if no scorer (e.g. a
    real web-search tool that returns no global index)."""
    if result is None or not result.hits:
        return -1.0
    if scorer is not None:
        return scorer.score(query)[result.hits[0][0]]
    return result.top_score


def _best(
    results: list[ToolResult], query: str, scorer: Optional[BM25] = None
) -> Optional[ToolResult]:
    """Re-rank completed speculative results against the (now full) query."""
    cands = [r for r in results if r.hits]
    if not cands:
        return None
    return max(cands, key=lambda r: _relevance(r, query, scorer))


# --------------------------------------------------------------------------- #
# Pipelines                                                                     #
# --------------------------------------------------------------------------- #
async def run_baseline(
    text: str,
    broker: ToolBroker,
    cfg: Config,
    *,
    input_source=None,
    on_word=None,
    on_fire=None,
    on_result=None,
    on_input_done=None,
    on_answer_ready=None,
) -> Metrics:
    """Fire-at-end-of-input: the conventional approach."""
    start = time.perf_counter()
    source = input_source if input_source is not None else _stream_words(text, cfg.words_per_sec)
    query = text
    async for partial in source:
        query = partial
        if on_word:
            on_word(partial)
    input_done = time.perf_counter()
    if on_input_done:
        on_input_done(input_done - start)
    if on_fire:
        on_fire(query, 1)
    await asyncio.sleep(cfg.query_gen_ms / 1000.0)
    result = await broker.search(query)
    if on_result:
        on_result(result, 1)
    await asyncio.sleep(cfg.fuse_ms / 1000.0)
    answer_ready = time.perf_counter()
    perceived_ms = (answer_ready - input_done) * 1000.0
    if on_answer_ready:
        on_answer_ready(perceived_ms)
    return Metrics(
        mode="baseline",
        perceived_ms=perceived_ms,
        wall_ms=(answer_ready - start) * 1000.0,
        calls_fired=1,
        calls_used=1,
        answer_doc=result.hits[0][0] if result.hits else -1,
        transport_overhead_ms=getattr(broker, "transport_overhead_ms", 0.0),
    )


async def run_streaming(
    text: str,
    broker: ToolBroker,
    cfg: Config,
    reflector: Reflector | None = None,
    *,
    input_source=None,
    on_word=None,
    on_fire=None,
    on_result=None,
    on_input_done=None,
    on_answer_ready=None,
) -> Metrics:
    """Speculative: Trigger (fixed-interval) + Threads (parallel) + Reflector."""
    reflector = reflector or Reflector()
    start = time.perf_counter()
    inflight: set[asyncio.Task] = set()
    results: list[ToolResult] = []
    fired = 0
    last_fire_words = 0
    seen_queries: set[str] = set()
    full_query = text

    async def dispatch(q: str):
        nonlocal fired
        fired += 1
        call_num = fired
        if on_fire:
            on_fire(q, call_num)
        await asyncio.sleep(cfg.query_gen_ms / 1000.0)
        r = await broker.search(q)
        results.append(r)
        if on_result:
            on_result(r, call_num)
        return r

    source = input_source if input_source is not None else _stream_words(text, cfg.words_per_sec)
    async for partial in source:
        full_query = partial
        if on_word:
            on_word(partial)
        n = len(partial.split())
        ready = n - last_fire_words >= cfg.trigger_interval_words
        room = len(inflight) < cfg.max_threads
        if ready and room and partial not in seen_queries:
            last_fire_words = n
            seen_queries.add(partial)
            t = asyncio.create_task(dispatch(partial))
            inflight.add(t)
            t.add_done_callback(inflight.discard)

    input_done = time.perf_counter()
    if on_input_done:
        on_input_done(input_done - start)
    scorer = getattr(broker, "scorer", None)

    def cancel_inflight():
        for t in list(inflight):
            t.cancel()

    # 1) Did a call that completed DURING input already give us a sufficient
    #    answer? If so, the tool latency was fully hidden -> answer immediately.
    answer = _best(results, full_query, scorer)
    if not reflector.sufficient(answer, full_query, scorer):
        # 2) Otherwise wait for in-flight calls, taking the first sufficient one.
        while inflight:
            await asyncio.wait(list(inflight), return_when=asyncio.FIRST_COMPLETED)
            answer = _best(results, full_query, scorer)
            if reflector.sufficient(answer, full_query, scorer):
                break
    # 3) Speculation still insufficient -> pay for one refining call on the full
    #    query. This is the failure mode the paper's better triggers minimize.
    if not reflector.sufficient(answer, full_query, scorer):
        cancel_inflight()
        answer = await dispatch(full_query)

    cancel_inflight()
    best = answer
    await asyncio.sleep(cfg.fuse_ms / 1000.0)
    answer_ready = time.perf_counter()

    perceived_ms = (answer_ready - input_done) * 1000.0
    if on_answer_ready:
        on_answer_ready(perceived_ms)

    used = 1
    return Metrics(
        mode="streaming",
        perceived_ms=perceived_ms,
        wall_ms=(answer_ready - start) * 1000.0,
        calls_fired=fired,
        calls_used=used,
        answer_doc=best.hits[0][0] if best and best.hits else -1,
        transport_overhead_ms=getattr(broker, "transport_overhead_ms", 0.0),
    )
