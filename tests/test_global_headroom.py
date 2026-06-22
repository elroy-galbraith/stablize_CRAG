import csv, os
from global_headroom import streamable_fraction, load_rows, sweep


def _write(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)


def test_load_rows_global_schema(tmp_path):
    p = os.path.join(tmp_path, "g.csv")
    _write(p, ["qid", "n_words", "t_suf_global", "phi_suf_global"], [
        {"qid": "a", "n_words": 10, "t_suf_global": 2, "phi_suf_global": 0.2},
        {"qid": "c", "n_words": 5, "t_suf_global": "", "phi_suf_global": ""},
    ])
    rows = load_rows(p)
    assert rows[0]["t_star"] == 2 and rows[0]["n_words"] == 10
    assert rows[1]["t_star"] is None


def test_load_rows_perq_schema_with_fallback(tmp_path):
    # per-question CRAG schema: prefer t_suf, fall back to t_sc
    p = os.path.join(tmp_path, "p.csv")
    _write(p, ["interaction_id", "n_words", "t_sc", "t_suf"], [
        {"interaction_id": "a", "n_words": 8, "t_sc": 3, "t_suf": 5},   # t_suf present -> 5
        {"interaction_id": "b", "n_words": 8, "t_sc": 4, "t_suf": ""},  # falls back to t_sc -> 4
        {"interaction_id": "c", "n_words": 8, "t_sc": "", "t_suf": ""}, # both empty -> None
    ])
    rows = load_rows(p, t_col="t_suf", fallback_col="t_sc")
    assert [r["t_star"] for r in rows] == [5, 4, None]


def test_streamable_fraction_basic(tmp_path):
    # n=10,t=2 -> H=min(600,2667)=600>=480 streamable; n=10,t=10 -> H=0 not; empty -> excluded
    p = os.path.join(tmp_path, "g.csv")
    _write(p, ["qid", "n_words", "t_suf_global"], [
        {"qid": "a", "n_words": 10, "t_suf_global": 2},
        {"qid": "b", "n_words": 10, "t_suf_global": 10},
        {"qid": "c", "n_words": 10, "t_suf_global": ""},
    ])
    s, d = streamable_fraction(load_rows(p), L_ms=600, delta_wps=3.0, theta=0.8)
    assert (s, d) == (1, 2)


def test_sweep_returns_one_cell_per_L(tmp_path):
    p = os.path.join(tmp_path, "g.csv")
    _write(p, ["qid", "n_words", "t_suf_global"], [
        {"qid": "a", "n_words": 12, "t_suf_global": 9},  # residual 3w=1000ms
    ])
    cells = sweep(load_rows(p), [600, 1500, 2500], delta_wps=3.0, theta=0.8)
    assert [c["L"] for c in cells] == [600, 1500, 2500]
    # H=min(L,1000); streamable iff 1000>=0.8L -> L<=1250: true@600, false@1500/2500
    assert cells[0]["frac"] == 1.0 and cells[1]["frac"] == 0.0 and cells[2]["frac"] == 0.0
