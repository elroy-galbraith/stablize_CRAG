"""
BM25-vs-dense comparison table (PROPOSAL_2 Phase 1 deliverable).

Reads two stabilization.summary.json files (produced by run_study.py under
--retriever bm25 and --retriever dense) and emits a side-by-side comparison as
markdown and LaTeX to stdout — the BM25/dense column for Table 1 / Figure 2.

The central Phase 1 question is whether the *ordering* of question types by
phi_suf agrees across retrievers (RQ4). The RQ4 block prints each retriever's
question-type ranking so divergence is obvious at a glance.

Usage:
    python compare_retrievers.py results/stabilization.summary.json \
                                  results/stabilization.dense.summary.json
Pure stdlib (json only); no run needed if the two summaries already exist.
"""
from __future__ import annotations

import json
import sys


def _g(d, *path, default=None):
    """Safe nested get."""
    for k in path:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def _fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def _label(summary, fallback):
    r = _g(summary, "params", "retriever")
    return r if r else fallback


def scalar_rows(a, b):
    """(metric label, value_a, value_b) for the headline metrics."""
    def frac(s, *p):
        v = _g(s, *p)
        return f"{100*v:.1f}%" if v is not None else "—"

    return [
        ("groundable rate", frac(a, "groundable", "frac"), frac(b, "groundable", "frac")),
        ("retrieved-gold rate", frac(a, "retrieved_gold", "frac"), frac(b, "retrieved_gold", "frac")),
        ("phi_sc mean", _fmt(_g(a, "rq1", "phi_sc", "mean")), _fmt(_g(b, "rq1", "phi_sc", "mean"))),
        ("phi_sc median", _fmt(_g(a, "rq1", "phi_sc", "median")), _fmt(_g(b, "rq1", "phi_sc", "median"))),
        ("phi_suf mean", _fmt(_g(a, "rq1", "phi_suf", "mean")), _fmt(_g(b, "rq1", "phi_suf", "mean"))),
        ("phi_suf median", _fmt(_g(a, "rq1", "phi_suf", "median")), _fmt(_g(b, "rq1", "phi_suf", "median"))),
        ("volatility mean", _fmt(_g(a, "rq1", "volatility_mean"), 2), _fmt(_g(b, "rq1", "volatility_mean"), 2)),
        ("RQ2 streamable frac", _fmt(_g(a, "rq2_configured", "frac")), _fmt(_g(b, "rq2_configured", "frac"))),
        ("RQ3 measured saved (ms)", _fmt(_g(a, "rq3", "measured_saved_ms_mean"), 1), _fmt(_g(b, "rq3", "measured_saved_ms_mean"), 1)),
        ("RQ3 H predicted (ms)", _fmt(_g(a, "rq3", "H_predicted_ms_mean"), 1), _fmt(_g(b, "rq3", "H_predicted_ms_mean"), 1)),
        ("RQ3 mis-fire rate", _fmt(_g(a, "rq3", "negative_saving_rate")), _fmt(_g(b, "rq3", "negative_saving_rate"))),
    ]


def rq4_ranking(summary):
    """[(question_type, mean_phi_suf)] sorted ascending (earliest-stabilizing first)."""
    by_type = _g(summary, "rq4_phi_suf_by_question_type", default={})
    items = [(qt, v.get("mean_phi_suf")) for qt, v in by_type.items()]
    return sorted(items, key=lambda kv: (kv[1] is None, kv[1]))


def print_markdown(la, lb, a, b):
    print(f"\n## RQ1/RQ2/RQ3 - {la} vs {lb}\n")
    print(f"| metric | {la} | {lb} |")
    print("|---|---|---|")
    for name, va, vb in scalar_rows(a, b):
        print(f"| {name} | {va} | {vb} |")

    print(f"\n## RQ4 - phi_suf by question_type (does the ordering agree?)\n")
    ra, rb = rq4_ranking(a), rq4_ranking(b)
    print(f"| rank | {la} (type -> phi_suf) | {lb} (type -> phi_suf) |")
    print("|---|---|---|")
    for i in range(max(len(ra), len(rb))):
        ca = f"{ra[i][0]} -> {_fmt(ra[i][1])}" if i < len(ra) else ""
        cb = f"{rb[i][0]} -> {_fmt(rb[i][1])}" if i < len(rb) else ""
        print(f"| {i+1} | {ca} | {cb} |")
    agree = [x[0] for x in ra] == [x[0] for x in rb]
    print(f"\n_Question-type ordering agrees across retrievers: **{agree}**_ "
          "(the Phase 1 entity-position account survives if True)._")


def print_latex(la, lb, a, b):
    print("\n% --- LaTeX (RQ1/RQ2/RQ3) ---")
    print("\\begin{tabular}{lrr}")
    print("\\toprule")
    print(f"Metric & {la} & {lb} \\\\")
    print("\\midrule")
    for name, va, vb in scalar_rows(a, b):
        safe = name.replace("_", "\\_")
        print(f"{safe} & {va} & {vb} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python compare_retrievers.py <bm25.summary.json> <dense.summary.json>")
    with open(sys.argv[1]) as f:
        a = json.load(f)
    with open(sys.argv[2]) as f:
        b = json.load(f)
    la = _label(a, "A")
    lb = _label(b, "B")
    print_markdown(la, lb, a, b)
    print_latex(la, lb, a, b)


if __name__ == "__main__":
    main()
