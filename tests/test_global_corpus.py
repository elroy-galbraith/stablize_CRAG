from global_corpus import corpus_row_to_text, qrels_to_dict


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
