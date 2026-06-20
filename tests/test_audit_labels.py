from audit_labels import sample_audit


def _rows(n):
    return [{"interaction_id": f"q{i}", "query": f"query {i}",
             "retrieved_gold": True, "t_suf": 2, "gold_passage": f"p{i}"} for i in range(n)]


def test_sample_audit_deterministic_and_sized():
    rows = _rows(50)
    a = sample_audit(rows, n=10, seed=0)
    b = sample_audit(rows, n=10, seed=0)
    assert len(a) == 10
    assert [r["interaction_id"] for r in a] == [r["interaction_id"] for r in b]
    assert all(r["is_answer_bearing"] == "" for r in a)
