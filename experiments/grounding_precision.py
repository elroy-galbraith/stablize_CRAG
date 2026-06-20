"""Grounding-precision correction for phi_suf (Paper 1 §5.1 / Limitations).

CRAG ships no gold-passage label; `crag.gold_passage_ids` derives d* by
string-matching the gold answer into passages. Short or ubiquitous answers
(a year, "0"/"1", "yes"/"no", or a term the whole page is about) match MANY
passages, so "gold in top-k" becomes near-unconditional and t_suf collapses to
1 for reasons unrelated to intent stabilization. This biases phi_suf LOW (early).

This script quantifies the bias and reports a precision-corrected phi_suf by
restricting to questions whose gold answer grounds *specifically* -- i.e. the
gold set covers only a small fraction of the question's passages. If a fraction
d of passages are labelled gold, a random top-k of size 3 hits one with
probability ~= 1-(1-d)^3 ~= 3d, so a density cap directly bounds the
"trivial hit" rate.

Inputs (no new retrieval -- reuses the canonical Paper 1 artifact):
  - results/stab_k3.csv  (t_suf, n_words per question; the central BM25 k=3 cell)
  - the CRAG bz2         (to recover gold-set size and answer token count)

Output: a sweep of phi_suf over density / gold-size caps, plus the headline
precision-corrected cell (density <= 0.05). Pure stdlib.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st

from crag import load_crag


def build_audit(stab_csv: str, data: str, split: int) -> list[dict]:
    """Join Paper 1's per-question t_suf with CRAG gold-set size / answer length."""
    tsuf, nwords = {}, {}
    with open(stab_csv) as f:
        for r in csv.DictReader(f):
            if r["retrieved_gold"] == "True" and r["t_suf"]:
                tsuf[r["interaction_id"]] = int(r["t_suf"])
                nwords[r["interaction_id"]] = int(r["n_words"])
    rows = []
    for ex in load_crag(data, split=split):
        if ex.interaction_id not in tsuf:
            continue
        n_pass = len(ex.passages)
        rows.append({
            "interaction_id": ex.interaction_id,
            "question_type": ex.question_type,
            "n_words": nwords[ex.interaction_id],
            "n_passages": n_pass,
            "gold_size": len(ex.gold),
            "gold_density": (len(ex.gold) / n_pass) if n_pass else 0.0,
            "ans_tokens": len(str(ex.answer).split()),
            "t_suf": tsuf[ex.interaction_id],
            "phi_suf": tsuf[ex.interaction_id] / max(nwords[ex.interaction_id], 1),
        })
    return rows


def cell(sub: list[dict]) -> dict:
    if not sub:
        return {"n": 0}
    phi = [r["phi_suf"] for r in sub]
    return {
        "n": len(sub),
        "phi_suf_mean": round(st.mean(phi), 4),
        "phi_suf_median": round(st.median(phi), 4),
        "t_suf_eq_1_rate": round(sum(1 for r in sub if r["t_suf"] == 1) / len(sub), 4),
        "t_suf_median": st.median([r["t_suf"] for r in sub]),
    }


def sweep(rows: list[dict]) -> dict:
    out = {"all": cell(rows), "by_density_cap": {}, "by_goldsize_cap": {},
           "headline_density_0.05": cell([r for r in rows if r["gold_density"] <= 0.05])}
    for cap in (0.10, 0.05, 0.02, 0.01):
        out["by_density_cap"][f"{cap}"] = cell([r for r in rows if r["gold_density"] <= cap])
    for k in (5, 3, 2, 1):
        out["by_goldsize_cap"][f"{k}"] = cell([r for r in rows if r["gold_size"] <= k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stab-csv", default="results/stab_k3.csv")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--audit-out", default="results/grounding_precision.split0.csv")
    ap.add_argument("--summary-out", default="results/grounding_precision.summary.json")
    args = ap.parse_args()

    rows = build_audit(args.stab_csv, args.data, args.split)
    with open(args.audit_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"params": {"stab_csv": args.stab_csv, "split": args.split,
                          "method": "gold-set density cap; lucky-hit ~= 3*density"},
               "sweep": sweep(rows)}
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    h = summary["sweep"]["headline_density_0.05"]
    a = summary["sweep"]["all"]
    print(f"all (n={a['n']}): phi_suf median={a['phi_suf_median']} mean={a['phi_suf_mean']} t_suf=1={a['t_suf_eq_1_rate']:.0%}")
    print(f"corrected density<=0.05 (n={h['n']}): phi_suf median={h['phi_suf_median']} "
          f"mean={h['phi_suf_mean']} t_suf=1={h['t_suf_eq_1_rate']:.0%}")
    print(f"wrote {args.audit_out}, {args.summary_out}")


if __name__ == "__main__":
    main()
