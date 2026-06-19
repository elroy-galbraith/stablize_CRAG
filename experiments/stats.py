"""
RQ4 inferential statistics for tool-intent stabilization (paper Sec. 5.2).

Reproduces, from a per-question stabilization CSV, every inferential number the
paper reports for RQ4:

  * Kruskal--Wallis omnibus test of phi_suf across question types
      - all types, and the subset of classes with n >= MIN_N (default 10)
  * rank-based effect size epsilon^2 for each test
  * Dunn's post-hoc (pairwise) with Holm correction
  * bootstrap (B resamples, fixed seed) 95% CIs on the per-type median phi_suf,
    which back the error intervals in the by-type figure

This is the artifact behind main.tex's claim that "the bootstrap confidence
intervals use 10,000 resamples at seed 0" and the Kruskal--Wallis / Dunn results
in results.tex. It is intentionally separate from the stdlib-only core pipeline:
it needs scipy/numpy, declared as the optional [stats] extra in pyproject.toml.

The test population is exactly the rows where phi_suf is defined, i.e. the
retrieved-gold subset (groundable AND gold present in some prefix's top-k); these
are the only rows for which sufficiency stabilization exists.

Usage:
  uv run --extra stats experiments/stats.py --csv results/stab_k3.csv
  uv run --extra stats experiments/stats.py --csv results/stab_k3.csv \
      --out results/rq4_stats.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from itertools import combinations

import numpy as np
from scipy import stats

MIN_N = 10          # "large" classes carrying the effect (results.tex Sec. 5.2)
N_BOOT = 10_000     # paper: "10,000 resamples"
SEED = 0            # paper: "at seed 0"
ALPHA = 0.05


def load_phi_suf_by_type(csv_path: str) -> "OrderedDict[str, list[float]]":
    """phi_suf grouped by question_type, over rows where phi_suf is defined."""
    groups: "OrderedDict[str, list[float]]" = OrderedDict()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("phi_suf") or "").strip()
            if not raw:                      # undefined -> not retrieved-gold
                continue
            qt = (row.get("question_type") or "unknown").strip()
            groups.setdefault(qt, []).append(float(raw))
    return groups


def epsilon_squared(H: float, n: int, k: int) -> dict:
    """Two reported conventions for the Kruskal--Wallis rank effect size.

    eps2_classic = H / (n - 1)            (Tomczak & Tomczak 2014)
    eps2_adj     = (H - k + 1) / (n - k)  (df-adjusted variant)
    """
    return {
        "eps2_classic": H / (n - 1),
        "eps2_adj": (H - k + 1) / (n - k),
    }


def kruskal_block(groups: "OrderedDict[str, list[float]]", label: str) -> dict:
    samples = [np.asarray(v, float) for v in groups.values()]
    k = len(samples)
    n = int(sum(len(s) for s in samples))
    H, p = stats.kruskal(*samples)
    eps = epsilon_squared(H, n, k)
    return {
        "label": label,
        "types": list(groups.keys()),
        "k": k,
        "df": k - 1,
        "n": n,
        "H": float(H),
        "p": float(p),
        **eps,
    }


def dunn_holm(groups: "OrderedDict[str, list[float]]") -> list[dict]:
    """Dunn's pairwise test on shared ranks, Holm-corrected over all pairs."""
    names = list(groups.keys())
    data = [np.asarray(groups[m], float) for m in names]
    sizes = {m: len(d) for m, d in zip(names, data)}
    allv = np.concatenate(data)
    N = allv.size
    ranks = stats.rankdata(allv)

    # mean rank per group
    mean_rank = {}
    idx = 0
    for m, d in zip(names, data):
        mean_rank[m] = ranks[idx:idx + len(d)].mean()
        idx += len(d)

    # tie correction term for the rank-sum variance
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts ** 3 - counts).sum()
    sigma_const = (N * (N + 1) / 12.0) - tie / (12.0 * (N - 1))

    pairs = []
    for a, b in combinations(names, 2):
        se = math.sqrt(sigma_const * (1.0 / sizes[a] + 1.0 / sizes[b]))
        z = float((mean_rank[a] - mean_rank[b]) / se) if se > 0 else 0.0
        p = float(2.0 * stats.norm.sf(abs(z)))
        pairs.append({"a": a, "b": b, "z": z, "p_raw": p})

    # Holm step-down
    order = sorted(range(len(pairs)), key=lambda i: pairs[i]["p_raw"])
    m = len(pairs)
    running = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * pairs[i]["p_raw"]
        running = max(running, adj)            # enforce monotonicity
        pairs[i]["p_holm"] = float(min(1.0, running))
        pairs[i]["significant"] = bool(pairs[i]["p_holm"] < ALPHA)
    return sorted(pairs, key=lambda d: d["p_holm"])


def bootstrap_median_ci(groups: "OrderedDict[str, list[float]]",
                        n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for m, vals in groups.items():
        a = np.asarray(vals, float)
        meds = np.empty(n_boot)
        for i in range(n_boot):
            meds[i] = np.median(rng.choice(a, size=a.size, replace=True))
        lo, hi = np.percentile(meds, [2.5, 97.5])
        out[m] = {
            "n": int(a.size),
            "median": float(np.median(a)),
            "ci95": [float(lo), float(hi)],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/stab_k3.csv",
                    help="per-question stabilization CSV (default: results/stab_k3.csv)")
    ap.add_argument("--min-n", type=int, default=MIN_N,
                    help="min group size for the 'large classes' KW test")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    groups = load_phi_suf_by_type(args.csv)
    big = OrderedDict((m, v) for m, v in groups.items() if len(v) >= args.min_n)

    kw_all = kruskal_block(groups, f"all types (k={len(groups)})")
    kw_big = kruskal_block(big, f"classes with n>={args.min_n} (k={len(big)})")
    dunn = dunn_holm(groups)
    boot = bootstrap_median_ci(groups, args.n_boot, args.seed)

    def show_kw(b):
        print(f"  {b['label']}: H={b['H']:.4f}  df={b['df']}  p={b['p']:.4f}  "
              f"n={b['n']}  eps2_classic={b['eps2_classic']:.4f}  "
              f"eps2_adj={b['eps2_adj']:.4f}")

    print(f"CSV: {args.csv}")
    print(f"phi_suf defined for n={kw_all['n']} questions across "
          f"{kw_all['k']} types\n")
    print("Kruskal--Wallis:")
    show_kw(kw_all)
    show_kw(kw_big)
    print("\nDunn's post-hoc (Holm), pairs sorted by p_holm:")
    for pr in dunn[:5]:
        flag = "*" if pr["significant"] else " "
        print(f"  {flag} {pr['a']:>16} vs {pr['b']:<16} "
              f"z={pr['z']:+.3f}  p_raw={pr['p_raw']:.4f}  p_holm={pr['p_holm']:.4f}")
    n_sig = sum(1 for p in dunn if p["significant"])
    print(f"  pairs significant at alpha={ALPHA}: {n_sig}/{len(dunn)}")
    print("\nBootstrap per-type median 95% CI "
          f"({args.n_boot} resamples, seed {args.seed}):")
    for m, b in sorted(boot.items(), key=lambda kv: kv[1]["median"]):
        print(f"  {m:>16}  n={b['n']:>3}  median={b['median']:.3f}  "
              f"CI=[{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]")

    if args.out:
        payload = {
            "params": {"csv": args.csv, "min_n": args.min_n,
                       "n_boot": args.n_boot, "seed": args.seed, "alpha": ALPHA},
            "kruskal_all": kw_all,
            "kruskal_large": kw_big,
            "dunn_holm": dunn,
            "bootstrap_median_ci": boot,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
