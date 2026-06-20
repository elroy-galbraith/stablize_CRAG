"""Clean vs string-grounded phi_suf on a benchmark with SHIPPED gold passages.
The clean arm uses ex.gold; the string arm re-derives gold via the CRAG matcher,
so the gap measures the over-grounding bias on independent labelled data.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from typing import Optional

from crag import CragExample, gold_passage_ids
from stabilization import stabilization


def dual_stab(ex: CragExample, top_k: int) -> Optional[dict]:
    clean = stabilization(ex.query, ex.passages, ex.gold, top_k=top_k)
    if clean is None:
        return None
    gold_string = gold_passage_ids(ex.answer, ex.alt_ans, ex.passages)
    strung = stabilization(ex.query, ex.passages, gold_string, top_k=top_k)
    return {
        "interaction_id": ex.interaction_id,
        "question_type": ex.question_type,
        "n_words": clean.n_words,
        "retrieved_gold_clean": clean.retrieved_gold,
        "t_suf_clean": clean.t_suf,
        "phi_suf_clean": clean.phi_suf,
        "retrieved_gold_string": strung.retrieved_gold if strung else False,
        "t_suf_string": strung.t_suf if strung else None,
        "phi_suf_string": strung.phi_suf if strung else None,
    }


def _cell(rows, gold_key, phi_key, t_key) -> dict:
    sub = [r for r in rows if r[gold_key] and r[phi_key] is not None]
    if not sub:
        return {"n": 0}
    phi = [r[phi_key] for r in sub]
    return {"n": len(sub),
            "phi_suf_mean": round(st.mean(phi), 4),
            "phi_suf_median": round(st.median(phi), 4),
            "t_suf_eq_1_rate": round(sum(1 for r in sub if r[t_key] == 1) / len(sub), 4)}


def summarize_dual(rows: list[dict]) -> dict:
    return {
        "clean": _cell(rows, "retrieved_gold_clean", "phi_suf_clean", "t_suf_clean"),
        "string": _cell(rows, "retrieved_gold_string", "phi_suf_string", "t_suf_string"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="hotpotqa", choices=["hotpotqa"])
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/hotpotqa/dual_stab.csv")
    ap.add_argument("--summary-out", default="results/hotpotqa/dual_stab.summary.json")
    args = ap.parse_args()

    from benchmarks import BENCHMARKS
    rows = [r for ex in BENCHMARKS[args.benchmark](args.split, limit=args.limit)
            if (r := dual_stab(ex, args.top_k))]
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(args.summary_out, "w") as f:
        json.dump({"params": {"benchmark": args.benchmark, "split": args.split, "top_k": args.top_k},
                   "n_questions": len(rows), "dual": summarize_dual(rows)}, f, indent=2)
    s = summarize_dual(rows)
    print(f"clean: {s['clean']}  | string: {s['string']}")


if __name__ == "__main__":
    main()
