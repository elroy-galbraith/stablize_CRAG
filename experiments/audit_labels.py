"""Sample retrieved-gold questions for a label-precision audit (spec §7)."""
from __future__ import annotations

import argparse
import csv
import random

from crag import load_crag
from stabilization import stabilization


def sample_audit(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    picked = rng.sample(rows, min(n, len(rows)))
    picked.sort(key=lambda r: r["interaction_id"])
    return [{"interaction_id": r["interaction_id"], "query": r["query"],
             "t_suf": r["t_suf"], "gold_passage": r["gold_passage"],
             "is_answer_bearing": ""} for r in picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", default="results/label_audit.csv")
    args = ap.parse_args()

    rows = []
    for ex in load_crag(args.data, split=args.split):
        s = stabilization(ex.query, ex.passages, ex.gold, top_k=args.top_k)
        if s is None or s.t_suf is None:
            continue
        gp = next(iter(ex.gold))
        rows.append({"interaction_id": ex.interaction_id, "query": ex.query,
                     "t_suf": s.t_suf, "gold_passage": ex.passages[gp]})
    audit = sample_audit(rows, args.n, args.seed)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["interaction_id", "query", "t_suf",
                                          "gold_passage", "is_answer_bearing"])
        w.writeheader()
        w.writerows(audit)
    print(f"wrote {len(audit)} audit rows -> {args.out}")


if __name__ == "__main__":
    main()
