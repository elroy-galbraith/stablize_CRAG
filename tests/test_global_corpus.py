from global_corpus import corpus_row_to_text, qrels_to_dict, GlobalBM25


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
    texts = ["bill gates co-founded microsoft",
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
