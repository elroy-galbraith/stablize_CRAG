from global_corpus import corpus_row_to_text, qrels_to_dict, GlobalBM25, global_t_suf, perq_t_suf


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
