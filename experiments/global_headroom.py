"""Streamable fraction under the global corpus (Paper 1 v2, RQ2 re-analysis).

Arithmetic over the committed global-corpus CSVs (columns:
qid,n_words,t_suf_global,phi_suf_global,t_suf_perq,phi_suf_perq). Reuses the
exact v1 hidden-latency bound so the global number is comparable to run_study's
per-question streamable fraction. No retrieval, no heavy deps.

Also supports the per-question CRAG v1 schema (interaction_id, t_suf, t_sc) via
the fallback_col parameter, enabling dual-schema L-sweep comparisons.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from stabilization import hidden_latency_ms


def load_rows(csv_path: str, t_col: str = "t_suf_global", fallback_col=None) -> list:
    """Load rows from a CSV, returning schema-neutral dicts with a 't_star' key.

    t_star = int(row[t_col]) if non-empty, else int(row[fallback_col]) if given
    and non-empty, else None. The per-row key is always 't_star' regardless of
    the source schema.
    """
    out = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t = r.get(t_col)
            if t in (None, "") and fallback_col:
                t = r.get(fallback_col)
            out.append({
                "qid": r.get("qid") or r.get("interaction_id"),
                "n_words": int(r["n_words"]),
                "t_star": int(t) if t not in (None, "") else None,
            })
    return out


def streamable_fraction(rows: list, L_ms: float, delta_wps: float, theta: float):
    """Return (streamable_count, denom) over rows with non-None t_star."""
    streamable = denom = 0
    for r in rows:
        if r["t_star"] is None:
            continue
        denom += 1
        if hidden_latency_ms(r["t_star"], r["n_words"], L_ms, delta_wps) >= theta * L_ms:
            streamable += 1
    return streamable, denom


def sweep(rows: list, Ls, delta_wps: float, theta: float) -> list:
    """Return one dict per L value with streamable count, denom, and fraction."""
    out = []
    for L in Ls:
        s, d = streamable_fraction(rows, L, delta_wps, theta)
        out.append({"L": L, "streamable": s, "denom": d, "frac": round(s / d, 4) if d else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--t-col", default="t_suf_global",
                    help="Column name for t_star (default: t_suf_global)")
    ap.add_argument("--fallback-col", default=None,
                    help="Fallback column when t_col is empty (e.g. t_sc for CRAG v1 schema)")
    ap.add_argument("--L", default="600,1500,2500",
                    help="Comma-separated tool latency values in ms (default: 600,1500,2500)")
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    Ls = [float(x) for x in args.L.split(",")]
    rows = load_rows(args.csv, t_col=args.t_col, fallback_col=args.fallback_col)
    cells = sweep(rows, Ls, args.delta, args.theta)

    label = args.label or os.path.basename(args.csv)
    summary = {
        "params": {
            "csv": args.csv,
            "label": label,
            "t_col": args.t_col,
            "fallback_col": args.fallback_col,
            "Ls": Ls,
            "delta": args.delta,
            "theta": args.theta,
        },
        "sweep": cells,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

    for c in cells:
        print(f"{label}  L={c['L']}: streamable {c['streamable']}/{c['denom']} = {c['frac']}"
              f"  (delta={args.delta}, theta={args.theta})")


if __name__ == "__main__":
    main()
