"""Streamable fraction under the global corpus (Paper 1 v2, RQ2 re-analysis).

Arithmetic over the committed global-corpus CSVs (columns:
qid,n_words,t_suf_global,phi_suf_global,t_suf_perq,phi_suf_perq). Reuses the
exact v1 hidden-latency bound so the global number is comparable to run_study's
per-question streamable fraction. No retrieval, no heavy deps.
"""
from __future__ import annotations

import csv
from typing import Optional

from stabilization import hidden_latency_ms


def load_rows(csv_path: str) -> list:
    out = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            t = r.get("t_suf_global")
            out.append({
                "qid": r.get("qid"),
                "n_words": int(r["n_words"]),
                "t_suf_global": int(t) if t not in (None, "") else None,
            })
    return out


def streamable_fraction(rows: list, L_ms: float, delta_wps: float, theta: float):
    streamable = denom = 0
    for r in rows:
        t = r["t_suf_global"]
        if t is None:
            continue
        denom += 1
        if hidden_latency_ms(t, r["n_words"], L_ms, delta_wps) >= theta * L_ms:
            streamable += 1
    return streamable, denom


def main():
    import argparse, json, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--L", type=float, default=600)
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    s, d = streamable_fraction(rows, args.L, args.delta, args.theta)
    frac = round(s / d, 4) if d else None
    label = args.label or os.path.basename(args.csv)
    summary = {"params": {"csv": args.csv, "label": label, "L": args.L,
                          "delta": args.delta, "theta": args.theta},
               "streamable": s, "denom": d, "streamable_fraction": frac}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"{label}: streamable {s}/{d} = {frac}  (L={args.L}, delta={args.delta}, theta={args.theta})")


if __name__ == "__main__":
    main()
