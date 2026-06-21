import numpy as np

from global_corpus import (
    corpus_row_to_text, qrels_to_dict, GlobalBM25, GlobalDense,
    global_t_suf, perq_t_suf,
    _build_corpus, _beir_repos,
)


def test_corpus_row_to_text():
    assert corpus_row_to_text({"_id": "d1", "title": "Paris", "text": "capital of France"}) == "Paris capital of France"
    assert corpus_row_to_text({"_id": "d2", "title": "", "text": "no title here"}) == "no title here"


def test_qrels_to_dict_filters_zero_scores():
    rows = [
        {"query-id": "q1", "corpus-id": "d1", "score": 1},
        {"query-id": "q1", "corpus-id": "d2", "score": 1},
        {"query-id": "q2", "corpus-id": "d3", "score": 0},  # dropped
        {"query-id": "q2", "corpus-id": "d4", "score": 1},
    ]
    assert qrels_to_dict(rows) == {"q1": {"d1", "d2"}, "q2": {"d4"}}


def _tiny_index():
    ids = ["d_gates", "d_paris", "d_moon"]
    texts = ["bill gates started microsoft",
             "paris is the capital of france",
             "the moon orbits the earth"]
    g = GlobalBM25()
    g.build(ids, texts)
    return g


def test_globalbm25_topk_ids_ranks_relevant_first():
    g = _tiny_index()
    res = g.topk_ids("who founded microsoft", k=2)
    assert res[0] == "d_gates"          # most relevant doc id first
    assert len(res) == 2


def test_globalbm25_empty_query():
    assert _tiny_index().topk_ids("", k=3) == []


def test_globalbm25_id_to_text():
    g = _tiny_index()
    assert g.id_to_text["d_paris"].startswith("paris")


class _FakeIndex:
    def __init__(self, by_t, id_to_text=None):
        self._by_t = by_t                      # {prefix_word_count: [doc_id, ...]}
        self.id_to_text = id_to_text or {}
    def topk_ids(self, query, k):
        return self._by_t.get(len(query.split()), [])[:k]


def test_global_t_suf_first_prefix_that_surfaces_gold():
    idx = _FakeIndex({1: ["a"], 2: ["b"], 3: ["g", "c"]})
    assert global_t_suf("w1 w2 w3", {"g"}, idx, k=2) == 3          # gold surfaces at prefix 3
    assert global_t_suf("w1 w2 w3", {"missing"}, idx, k=2) is None  # gold never returned -> None
    assert global_t_suf("w1 w2 w3", {"c"}, idx, k=1) is None        # at t=3, k=1 returns only "g"; gold "c" not hit


def test_perq_t_suf_pool_is_easier_or_equal():
    g = _tiny_index()
    tg = global_t_suf("who founded microsoft", {"d_gates"}, g, k=1)
    tp = perq_t_suf("who founded microsoft", {"d_gates"}, g, k=1, n_pool=3)
    assert tp is not None and (tg is None or tp <= tg)


def test_summarize_two_arms():
    from global_corpus import summarize
    rows = [
        {"t_suf_global": 4, "phi_suf_global": 0.4, "t_suf_perq": 1, "phi_suf_perq": 0.1},
        {"t_suf_global": 6, "phi_suf_global": 0.6, "t_suf_perq": 1, "phi_suf_perq": 0.1},
        {"t_suf_global": None, "phi_suf_global": None, "t_suf_perq": 2, "phi_suf_perq": 0.2},
    ]
    s = summarize(rows)
    assert s["global"]["n"] == 2
    assert s["global"]["phi_suf_median"] == 0.5
    assert s["perq"]["n"] == 3
    assert s["perq"]["t_suf_eq_1_rate"] == round(2 / 3, 4)


# ---------------------------------------------------------------------------
# _build_corpus tests
# ---------------------------------------------------------------------------

def _fake_corpus():
    """6-row fake corpus: g1, g2 (gold) + d1..d4 (non-gold)."""
    return [
        {"_id": "g1", "title": "Gold One", "text": "alpha"},
        {"_id": "g2", "title": "Gold Two", "text": "beta"},
        {"_id": "d1", "title": "Dist One", "text": "gamma"},
        {"_id": "d2", "title": "Dist Two", "text": "delta"},
        {"_id": "d3", "title": "Dist Three", "text": "epsilon"},
        {"_id": "d4", "title": "Dist Four", "text": "zeta"},
    ]


def test_build_corpus_gold_always_present():
    ids, texts = _build_corpus(_fake_corpus(), {"g1", "g2"}, n_distractors=2, seed=0)
    assert "g1" in ids
    assert "g2" in ids


def test_build_corpus_distractor_count():
    ids, texts = _build_corpus(_fake_corpus(), {"g1", "g2"}, n_distractors=2, seed=0)
    assert len(ids) == 4  # 2 gold + 2 distractors
    assert len(texts) == 4


def test_build_corpus_only_distractors_from_non_gold():
    ids, texts = _build_corpus(_fake_corpus(), {"g1", "g2"}, n_distractors=2, seed=0)
    non_gold = [i for i in ids if i not in {"g1", "g2"}]
    assert len(non_gold) == 2
    assert all(i in {"d1", "d2", "d3", "d4"} for i in non_gold)


def test_build_corpus_deterministic():
    ids_a, _ = _build_corpus(_fake_corpus(), {"g1", "g2"}, n_distractors=2, seed=0)
    ids_b, _ = _build_corpus(_fake_corpus(), {"g1", "g2"}, n_distractors=2, seed=0)
    assert ids_a == ids_b


def test_build_corpus_keep_all_when_none():
    ids, texts = _build_corpus(_fake_corpus(), {"g1", "g2"}, n_distractors=None, seed=0)
    assert set(ids) == {"g1", "g2", "d1", "d2", "d3", "d4"}
    assert len(ids) == 6


def test_build_corpus_texts_match_ids():
    """corpus_row_to_text should be applied consistently."""
    ids, texts = _build_corpus(_fake_corpus(), {"g1"}, n_distractors=None, seed=0)
    idx = ids.index("g1")
    assert texts[idx] == "Gold One alpha"


# ---------------------------------------------------------------------------
# _beir_repos tests
# ---------------------------------------------------------------------------

def test_beir_repos_nq():
    assert _beir_repos("nq") == ("BeIR/nq", "BeIR/nq-qrels", "test")


def test_beir_repos_fiqa():
    assert _beir_repos("fiqa") == ("BeIR/fiqa", "BeIR/fiqa-qrels", "test")


def test_beir_repos_hotpotqa():
    assert _beir_repos("hotpotqa") == ("BeIR/hotpotqa", "BeIR/hotpotqa-qrels", "test")


def test_beir_repos_scifact():
    assert _beir_repos("scifact") == ("BeIR/scifact", "BeIR/scifact-qrels", "test")


def test_beir_repos_unknown():
    import pytest
    with pytest.raises(KeyError):
        _beir_repos("unknown_dataset")


# ---------------------------------------------------------------------------
# GlobalDense tests (no model download — inject fake embeddings)
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal SentenceTransformer stub."""

    def __init__(self, response: np.ndarray):
        self._response = response  # shape (1, d)

    def encode(self, sentences, normalize_embeddings=True, convert_to_numpy=True, **kw):
        return self._response


def test_global_dense_topk_ids_ranking():
    g = GlobalDense()
    g.ids = ["a", "b", "c"]
    # qv=[1,0] → sims: a=1.0, b=0.0, c=0.9 → top-2 = ["a","c"]
    g._emb = np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=float)
    g._model = _FakeModel(np.array([[1.0, 0.0]]))
    result = g.topk_ids("any query", 2)
    assert result == ["a", "c"]


def test_global_dense_topk_ids_empty_query():
    g = GlobalDense()
    g.ids = ["a", "b"]
    g._emb = np.array([[1, 0], [0, 1]], dtype=float)
    g._model = _FakeModel(np.array([[1.0, 0.0]]))
    assert g.topk_ids("", 2) == []
    assert g.topk_ids("   ", 2) == []


def test_global_dense_topk_ids_k_clipped():
    """k larger than corpus size should not crash."""
    g = GlobalDense()
    g.ids = ["a"]
    g._emb = np.array([[1, 0]], dtype=float)
    g._model = _FakeModel(np.array([[1.0, 0.0]]))
    result = g.topk_ids("hello", 10)
    assert result == ["a"]
