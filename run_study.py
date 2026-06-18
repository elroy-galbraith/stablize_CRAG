"""
RQ1-RQ3 driver for the tool-intent stabilization study.

Usage:
  python3 run_study.py --data crag_task_1_and_2_dev_v4.jsonl.bz2 --split 0 --out stab.csv
  python3 run_study.py --data crag_fixture.jsonl.bz2 --split 0 --latency-n 5   # smoke test

No third-party deps required (BM25 is pure-python). bs4 improves HTML cleaning
if installed; matplotlib enables --plot. CRAG data is NOT bundled — download
crag_task_1_and_2_dev_v4.jsonl.bz2 from the facebookresearch/CRAG repo.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict

from crag import load_crag
from stabilization import stabilization, hidden_latency_ms

FIELDS = [
    "interaction_id", "domain", "question_type", "static_or_dynamic",
    "n_words", "n_passages", "groundable", "retrieved_gold",
    "t_sc", "phi_sc", "t_suf", "phi_suf", "volatility",
]
LATENCY_FIELDS = ["interaction_id", "question_type", "n_words", "t_star",
                  "measured_saved_ms", "H_predicted_ms"]

# PROPOSAL §7 factor grid for the RQ2 streamable surface (recomputed from t*).
RQ2_GRID_L = [100.0, 300.0, 600.0, 1000.0]
RQ2_GRID_DELTA = [2.0, 3.0, 4.0]
RQ2_GRID_THETA = [0.5, 0.8, 1.0]


def _streamable_fraction(rows, L_ms, delta_wps, theta):
    """RQ2: share of questions whose hideable latency H >= theta*L. Uses t_suf,
    falls back to t_sc. Returns (streamable, denom)."""
    streamable = 0
    denom = 0
    for r in rows:
        t_star = r["t_suf"] if r["t_suf"] else r["t_sc"]
        if t_star is None:
            continue
        denom += 1
        H = hidden_latency_ms(t_star, r["n_words"], L_ms, delta_wps)
        if H >= theta * L_ms:
            streamable += 1
    return streamable, denom


def _stats(vals):
    return {"n": len(vals), "mean": st.mean(vals), "median": st.median(vals)} if vals else None


def summarize(rows, L_ms, delta_wps, theta):
    """Print the RQ1/RQ2/RQ4 summary AND return it as a structured dict."""
    def col(name, pred=lambda r: True):
        return [r[name] for r in rows if r[name] is not None and pred(r)]

    n = len(rows)
    groundable = [r for r in rows if r["groundable"]]
    retrieved = [r for r in rows if r["retrieved_gold"]]
    print(f"\n{'='*72}\nSTABILIZATION SUMMARY  (n={n} questions)\n{'='*72}")
    print(f"Groundable (gold answer found in pages): {len(groundable)}/{n} "
          f"({100*len(groundable)/max(n,1):.1f}%)")
    print(f"Gold retrieved in top-k at some prefix:  {len(retrieved)}/{n} "
          f"({100*len(retrieved)/max(n,1):.1f}%)")

    phi_sc = col("phi_sc")
    if phi_sc:
        print(f"\n[RQ1] phi_sc  (self-consistency): mean {st.mean(phi_sc):.3f}  "
              f"median {st.median(phi_sc):.3f}")
    phi_suf = [r["phi_suf"] for r in retrieved]
    if phi_suf:
        print(f"[RQ1] phi_suf (sufficiency, retrieved only): mean {st.mean(phi_suf):.3f}  "
              f"median {st.median(phi_suf):.3f}")
    vol = col("volatility")
    if vol:
        print(f"[RQ1] volatility: mean {st.mean(vol):.2f}  "
              f"(share with V>0: {100*sum(1 for v in vol if v>0)/len(vol):.1f}%)")

    # RQ4: per-question-type phi_suf
    print(f"\n[RQ4] phi_suf by question_type (retrieved-gold questions):")
    by_type = defaultdict(list)
    for r in retrieved:
        by_type[r["question_type"]].append(r["phi_suf"])
    rq4 = {}
    for qt in sorted(by_type, key=lambda k: st.mean(by_type[k])):
        v = by_type[qt]
        rq4[qt] = {"n": len(v), "mean_phi_suf": st.mean(v)}
        print(f"   {qt:<18} n={len(v):<4} mean phi_suf={st.mean(v):.3f}")

    # RQ2: streamable fraction at the configured (L, delta, theta) ...
    streamable, denom = _streamable_fraction(rows, L_ms, delta_wps, theta)
    print(f"\n[RQ2] streamable fraction at L={L_ms:.0f}ms, delta={delta_wps} w/s, "
          f"theta={theta}: {streamable}/{denom} ({100*streamable/max(denom,1):.1f}%)")
    print("      (streamable = at least theta*L of tool latency hideable behind input)")

    # ... and the full PROPOSAL §7 grid (cheap: arithmetic over already-computed t*).
    rq2_grid = []
    for L in RQ2_GRID_L:
        for d in RQ2_GRID_DELTA:
            for th in RQ2_GRID_THETA:
                s, dn = _streamable_fraction(rows, L, d, th)
                rq2_grid.append({"L_ms": L, "delta_wps": d, "theta": th,
                                 "streamable": s, "denom": dn,
                                 "frac": s / dn if dn else None})

    return {
        "n_questions": n,
        "groundable": {"count": len(groundable), "frac": len(groundable) / max(n, 1)},
        "retrieved_gold": {"count": len(retrieved), "frac": len(retrieved) / max(n, 1)},
        "rq1": {
            "phi_sc": _stats(phi_sc),
            "phi_suf": _stats(phi_suf),
            "volatility_mean": st.mean(vol) if vol else None,
            "volatility_share_gt0": (sum(1 for v in vol if v > 0) / len(vol)) if vol else None,
        },
        "rq4_phi_suf_by_question_type": rq4,
        "rq2_configured": {"L_ms": L_ms, "delta_wps": delta_wps, "theta": theta,
                           "streamable": streamable, "denom": denom,
                           "frac": streamable / denom if denom else None},
        "rq2_grid": rq2_grid,
    }


async def validate_latency(examples, L_ms, delta_wps, n_max, out_csv, pop="suf"):
    """RQ3: compare measured perceived latency (baseline vs streaming) to the
    H-bound, on a small subset, using the existing async harness. Writes a
    per-question CSV and returns a structured summary dict (None if no rows).

    `pop` selects the population: "suf" replays retrieved-gold questions (t* =
    t_suf, the favorable early-stabilizing slice); "sc" replays the fallback
    majority that has no retrieved gold (t* = t_sc), which is the population most
    exposed to trigger mis-fires — used to estimate the downside rate H ignores."""
    from streaming_rag import Config, DirectRetrievalBroker, run_baseline, run_streaming

    cfg = Config(words_per_sec=delta_wps, trigger_interval_words=3, max_threads=4)
    rows = []
    for ex in examples:
        if len(rows) >= n_max:
            break
        if pop == "suf" and not ex.retrieved_gold_stab:   # attached below
            continue
        if pop == "sc" and ex.retrieved_gold_stab:        # want the fallback majority
            continue
        docs = ex.passages
        b = await run_baseline(ex.query, DirectRetrievalBroker(docs, exec_latency_ms=L_ms), cfg)
        s = await run_streaming(ex.query, DirectRetrievalBroker(docs, exec_latency_ms=L_ms), cfg)
        t_star = ex.stab.t_suf or ex.stab.t_sc
        rows.append({
            "interaction_id": ex.interaction_id,
            "question_type": ex.question_type,
            "n_words": ex.stab.n_words,
            "t_star": t_star,
            "measured_saved_ms": round(b.perceived_ms - s.perceived_ms, 2),
            "H_predicted_ms": round(hidden_latency_ms(t_star, ex.stab.n_words, L_ms, delta_wps), 2),
        })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LATENCY_FIELDS)
        w.writeheader()
        w.writerows(rows)

    if not rows:
        print("\n[RQ3] no eligible questions for latency validation in this subset.")
        print(f"      (wrote empty {out_csv})")
        return None

    saved = [r["measured_saved_ms"] for r in rows]
    predicted = [r["H_predicted_ms"] for r in rows]
    n_neg = sum(1 for v in saved if v < 0)
    print(f"\n[RQ3] latency validation on {len(rows)} questions (pop={pop}) -> {out_csv}")
    print(f"   measured perceived-latency saved: mean {st.mean(saved):.1f}ms")
    print(f"   H-bound predicted saving:         mean {st.mean(predicted):.1f}ms")
    print(f"   net-negative-saving (mis-fire) rate: {n_neg}/{len(rows)} "
          f"({100*n_neg/len(rows):.1f}%)")
    return {
        "n": len(rows), "pop": pop,
        "L_ms": L_ms, "delta_wps": delta_wps,
        "measured_saved_ms_mean": st.mean(saved),
        "H_predicted_ms_mean": st.mean(predicted),
        "negative_saving_count": n_neg,
        "negative_saving_rate": n_neg / len(rows),
        "csv": out_csv,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--chunk-words", type=int, default=120)
    ap.add_argument("--out", default="stabilization.csv")
    ap.add_argument("--L", type=float, default=600.0, help="tool latency ms (RQ2/RQ3)")
    ap.add_argument("--delta", type=float, default=3.0, help="input cadence words/sec")
    ap.add_argument("--theta", type=float, default=0.8, help="streamable coverage threshold")
    ap.add_argument("--latency-n", type=int, default=0, help="RQ3 subset size (0=skip)")
    ap.add_argument("--latency-pop", choices=["suf", "sc"], default="suf",
                    help="RQ3 population: 'suf'=retrieved-gold (t_suf), "
                         "'sc'=fallback majority (t_sc, no retrieved gold)")
    ap.add_argument("--grounding", choices=["exact", "fuzzy"], default="exact",
                    help="d* grounding: exact substring or fuzzy bag-of-content-token")
    ap.add_argument("--summary-json", default="stabilization.summary.json",
                    help="aggregate RQ1/RQ2/RQ4(+grid)/RQ3 results JSON for the report")
    ap.add_argument("--latency-csv", default="latency_validation.csv",
                    help="per-question RQ3 measured-vs-predicted rows")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-out", default="phi_distribution.png")
    args = ap.parse_args()

    rows = []
    keep_for_latency = []
    for ex in load_crag(args.data, split=args.split, limit=args.limit,
                        chunk_words=args.chunk_words, grounding=args.grounding):
        stab = stabilization(ex.query, ex.passages, ex.gold, top_k=args.top_k)
        if stab is None:
            continue
        rows.append({
            "interaction_id": ex.interaction_id, "domain": ex.domain,
            "question_type": ex.question_type, "static_or_dynamic": ex.static_or_dynamic,
            "n_words": stab.n_words, "n_passages": stab.n_passages,
            "groundable": ex.groundable, "retrieved_gold": stab.retrieved_gold,
            "t_sc": stab.t_sc, "phi_sc": round(stab.phi_sc, 4),
            "t_suf": stab.t_suf, "phi_suf": round(stab.phi_suf, 4) if stab.phi_suf else None,
            "volatility": stab.volatility,
        })
        if args.latency_n:
            ex.stab = stab
            ex.retrieved_gold_stab = stab.retrieved_gold
            keep_for_latency.append(ex)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")

    summary = summarize(rows, args.L, args.delta, args.theta)

    # Provenance so the JSON is self-describing for the report / reproducibility.
    summary["params"] = {
        "data": args.data, "split": args.split, "limit": args.limit,
        "top_k": args.top_k, "chunk_words": args.chunk_words,
        "L_ms": args.L, "delta_wps": args.delta, "theta": args.theta,
        "per_question_csv": args.out,
    }
    summary["rq3"] = None

    if args.latency_n:
        import asyncio
        summary["rq3"] = asyncio.run(
            validate_latency(keep_for_latency, args.L, args.delta, args.latency_n,
                             args.latency_csv, pop=args.latency_pop)
        )

    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote aggregate summary -> {args.summary_json}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            phi = [r["phi_suf"] for r in rows if r["phi_suf"] is not None]
            if phi:
                plt.figure(figsize=(6, 4))
                plt.hist(phi, bins=20, color="#1F3B57", edgecolor="white")
                plt.xlabel("phi_suf  (sufficiency stabilization fraction)")
                plt.ylabel("questions")
                plt.title("Tool-intent stabilization (lower = earlier = more hideable)")
                plt.tight_layout()
                plt.savefig(args.plot_out, dpi=130)
                print(f"Wrote {args.plot_out}")
        except Exception as e:
            print(f"[plot] skipped ({e}); pip install matplotlib to enable.")


if __name__ == "__main__":
    main()
