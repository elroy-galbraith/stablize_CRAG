from benchmark_stats import dual_stab, summarize_dual
from crag import CragExample


def test_dual_stab_clean_vs_string():
    # clean gold = passage 0 (shipped, with answer "planet earth orbits the sun").
    # string grounding will match the answer against both passages and derive gold.
    # The key is that both clean and string must produce non-None t_suf values.
    ex = CragExample("q1", "does earth orbit the sun", "planet earth orbits the sun", [], "", "simple", "", 0,
                     passages=["the planet earth orbits the sun",
                               "earth has 1 moon called luna"],
                     gold={0})
    r = dual_stab(ex, top_k=1)
    assert r is not None
    assert r["t_suf_clean"] is not None
    # string grounding marks any passage containing "planet earth orbits the sun" as gold
    # (passage 0); both are defined here
    assert r["t_suf_string"] is not None
    assert r["n_words"] == 5


def test_summarize_dual_reports_both():
    rows = [
        {"retrieved_gold_clean": True, "phi_suf_clean": 0.4, "t_suf_clean": 2,
         "retrieved_gold_string": True, "phi_suf_string": 0.1, "t_suf_string": 1,
         "n_words": 5, "question_type": "comparison"},
        {"retrieved_gold_clean": True, "phi_suf_clean": 0.5, "t_suf_clean": 3,
         "retrieved_gold_string": True, "phi_suf_string": 0.17, "t_suf_string": 1,
         "n_words": 6, "question_type": "bridge"},
    ]
    s = summarize_dual(rows)
    assert s["clean"]["phi_suf_median"] == 0.45
    assert s["string"]["phi_suf_median"] == 0.135
    assert s["string"]["t_suf_eq_1_rate"] == 1.0
    assert s["clean"]["t_suf_eq_1_rate"] == 0.0
