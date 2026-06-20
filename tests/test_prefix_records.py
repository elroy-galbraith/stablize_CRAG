from stabilization import prefix_records, stabilization


def test_prefix_records_shape_and_monotonic_prefixes():
    passages = ["alpha beta", "gamma delta", "alpha gamma epsilon"]
    seq, n = prefix_records("alpha gamma epsilon", passages, top_k=2)
    assert n == 3
    assert len(seq) == 3
    for top1, ids in seq:
        assert isinstance(top1, int)
        assert isinstance(ids, set)
        assert len(ids) <= 2


def test_prefix_records_empty():
    assert prefix_records("", ["x"], top_k=3) == ([], 0)
    assert prefix_records("q", [], top_k=3) == ([], 0)


def test_stabilization_unchanged_after_refactor():
    passages = ["the eiffel tower is in paris", "paris is in france", "berlin germany"]
    s = stabilization("where is the eiffel tower", passages, gold={0}, top_k=2)
    assert s is not None
    assert s.n_words == 5
    assert s.t_suf is not None and 1 <= s.t_suf <= 5
    assert s.retrieved_gold is True
