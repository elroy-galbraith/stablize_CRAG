"""
Dense retriever condition (PROPOSAL §6 H4 / PROPOSAL_2 Phase 1).

A drop-in alternative to the vendored `BM25` for the stabilization sweep and the
RQ3 latency harness. The point of Phase 1: BM25 fires early on keyword anchors, so
low phi_suf might be a retriever artifact rather than a query property. Re-running
the sweep under a dense (sentence-embedding) retriever tests whether the
early-stabilization finding — and the question-type ordering — survives.

Design:
  - `DenseRetriever` mirrors the `BM25` interface (`score`, `topk`) so it is a
    duck-typed drop-in everywhere BM25 is used, plus a batched `prefix_topk`
    fast path so a question's whole prefix sweep is one encode call.
  - `DenseRetrievalBroker` subclasses the vendored `DirectRetrievalBroker` and
    only swaps the retriever, so the async RQ3 pipeline (search + `.scorer`
    re-rank) runs under dense retrieval with zero edits to streaming_rag.py.

Heavy deps (`sentence-transformers`, transitively torch) are imported lazily so
the BM25 core stays zero-dependency. Install with the `dense` extra:
    uv pip install -e ".[dense]"
"""
from __future__ import annotations

from typing import Optional

from streaming_rag import DirectRetrievalBroker

DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_dense_model(name: str = DEFAULT_DENSE_MODEL):
    """Load a SentenceTransformer once, to be reused across all questions.

    The first call downloads the model weights (~80MB for all-MiniLM-L6-v2) from
    the HuggingFace hub and needs network access. Subsequent calls hit the local
    cache. Raises a clear error if the optional `dense` extra is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - environment guard
        raise ImportError(
            "Dense retrieval needs sentence-transformers. Install the extra:\n"
            '    uv pip install -e ".[dense]"   (or: uv run --extra dense ...)'
        ) from e
    return SentenceTransformer(name)


class DenseRetriever:
    """Cosine-similarity retriever over a question's passages.

    Mirrors `BM25`'s interface: `score(query) -> list[float]` (per-doc, used by the
    re-rank `.scorer`) and `topk(query, k) -> list[(id, score)]` (the contract
    `stabilization` and the broker rely on). Embeddings are L2-normalized so cosine
    similarity is a plain dot product.
    """

    def __init__(self, passages: list[str], model):
        import numpy as np

        self.passages = passages
        self.model = model
        # (N, d) normalized passage matrix — encoded once per question.
        self.emb = np.asarray(
            model.encode(passages, normalize_embeddings=True, convert_to_numpy=True),
            dtype="float32",
        )

    def _sims(self, queries: list[str]):
        """(len(queries), N) cosine-similarity matrix."""
        import numpy as np

        q = np.asarray(
            self.model.encode(queries, normalize_embeddings=True, convert_to_numpy=True),
            dtype="float32",
        )
        if q.ndim == 1:
            q = q[None, :]
        return q @ self.emb.T

    def score(self, query: str) -> list[float]:
        return self._sims([query])[0].tolist()

    def topk(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        return self._rank(self._sims([query])[0], k)

    def prefix_topk(self, words: list[str], top_k: int):
        """Batched prefix sweep: encode every prefix q[1:t] in ONE encode call,
        one matmul against the passage matrix, then top-k per row. Returns the
        `(top1_id, set_of_topk_ids)` sequence `_prefix_sequence` expects."""
        prefixes = [" ".join(words[:t]) for t in range(1, len(words) + 1)]
        sims = self._sims(prefixes)
        seq = []
        for row in sims:
            ranked = self._rank(row, top_k)
            ids = [i for i, _ in ranked]
            seq.append((ids[0] if ids else -1, set(ids)))
        return seq

    @staticmethod
    def _rank(row, k: int) -> list[tuple[int, float]]:
        import numpy as np

        order = np.argsort(-row, kind="stable")[:k]
        return [(int(i), float(row[i])) for i in order]


class DenseRetrievalBroker(DirectRetrievalBroker):
    """RQ3 broker backed by `DenseRetriever`. Subclasses the vendored
    `DirectRetrievalBroker` and only swaps the retriever stored in `self.bm25`
    (the duck-typed scorer slot); `.scorer` and `.search` are inherited unchanged,
    so the streaming pipeline re-ranks and searches through dense retrieval."""

    def __init__(
        self,
        corpus: list[str],
        model,
        exec_latency_ms: float = 400.0,
        transport_overhead_ms: float = 0.0,
        top_k: int = 3,
    ):
        self.corpus = corpus
        self.bm25 = DenseRetriever(corpus, model)  # duck-typed scorer/search backend
        self.exec_latency_ms = exec_latency_ms
        self.transport_overhead_ms = transport_overhead_ms
        self.top_k = top_k
