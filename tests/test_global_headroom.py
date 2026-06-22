# tests/test_global_headroom.py
import csv, os
from global_headroom import streamable_fraction, load_rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "n_words", "t_suf_global", "phi_suf_global"])
        w.writeheader()
        w.writerows(rows)


def test_streamable_fraction_basic(tmp_path):
    # row1: n=10,t=2 -> residual=(8/3)*1000=2667ms, H=min(600,2667)=600 >= 0.8*600=480 -> streamable
    # row2: n=10,t=10 -> residual=0, H=0 -> not streamable
    # row3: t_suf_global empty -> excluded from denom entirely
    p = os.path.join(tmp_path, "g.csv")
    _write_csv(p, [
        {"qid": "a", "n_words": 10, "t_suf_global": 2, "phi_suf_global": 0.2},
        {"qid": "b", "n_words": 10, "t_suf_global": 10, "phi_suf_global": 1.0},
        {"qid": "c", "n_words": 10, "t_suf_global": "", "phi_suf_global": ""},
    ])
    rows = load_rows(p)
    s, d = streamable_fraction(rows, L_ms=600, delta_wps=3.0, theta=0.8)
    assert (s, d) == (1, 2)


def test_load_rows_parses_none(tmp_path):
    p = os.path.join(tmp_path, "g.csv")
    _write_csv(p, [{"qid": "c", "n_words": 5, "t_suf_global": "", "phi_suf_global": ""}])
    rows = load_rows(p)
    assert rows[0]["t_suf_global"] is None
    assert rows[0]["n_words"] == 5
