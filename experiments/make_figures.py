"""
Build the paper figures from the run artifacts in results/.
Reads the per-question CSVs and summary JSONs (no re-run, no CRAG data needed).
Run: uv run --extra plot python3 make_figures.py
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(_ROOT, "results")
FIGS = os.path.join(_ROOT, "paper", "figures")
NAVY = "#1F3B57"
RUST = "#B4543A"
os.makedirs(FIGS, exist_ok=True)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    return float(x) if x not in (None, "", "None") else None


# --------------------------------------------------------------------------- #
# Fig 1: phi distributions (sufficiency vs self-consistency), k=3              #
# --------------------------------------------------------------------------- #
def fig_phi_distribution(rows):
    phi_suf = [fnum(r["phi_suf"]) for r in rows if fnum(r["phi_suf"]) is not None]
    phi_sc = [fnum(r["phi_sc"]) for r in rows if fnum(r["phi_sc"]) is not None]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = [i / 20 for i in range(21)]
    ax.hist(phi_sc, bins=bins, color="#9bb4c4", edgecolor="white",
            label=r"$\phi_{sc}$ (self-consistency)")
    ax.hist(phi_suf, bins=bins, color=NAVY, edgecolor="white", alpha=0.85,
            label=r"$\phi_{suf}$ (sufficiency)")
    ax.set_xlabel("stabilization fraction  $\\phi = t^\\star/n$")
    ax.set_ylabel("questions")
    ax.set_title("Tool-intent stabilization (lower = earlier = more hideable)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGS}/phi_distribution.png", dpi=140)
    plt.close(fig)
    print("wrote phi_distribution.png")


# --------------------------------------------------------------------------- #
# Fig 2: phi_suf by question type (k=3), horizontal boxplot, sorted by median  #
# --------------------------------------------------------------------------- #
def fig_phi_by_type(rows):
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in rows:
        v = fnum(r.get("phi_suf"))
        if v is not None:
            by_type[r["question_type"]].append(v)
    items = sorted(by_type.items(), key=lambda kv: sorted(kv[1])[len(kv[1]) // 2])
    labels = [k.replace("simple_w_condition", "simple w/\ncondition").replace("_", "\n") for k, _ in items]
    data = [v for _, v in items]
    counts = [len(v) for _, v in items]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True,
                    boxprops=dict(facecolor="#9bb4c4", color=NAVY),
                    medianprops=dict(color=NAVY, linewidth=2),
                    whiskerprops=dict(color=NAVY),
                    capprops=dict(color=NAVY),
                    flierprops=dict(marker="o", markerfacecolor=RUST,
                                   markersize=3, alpha=0.6, linestyle="none"))
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=9)
    for i, (c, d) in enumerate(zip(counts, data), 1):
        ax.text(max(d) + 0.025, i, f"n={c}", va="center", fontsize=8)
    ax.set_xlabel(r"$\phi_{\mathrm{suf}}$  (stabilization fraction,  lower = earlier)")
    ax.set_title("Sufficiency stabilization by question type ($k{=}3$)")
    ax.set_xlim(0, 1.18)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/phi_by_type.png", dpi=140)
    plt.close(fig)
    print("wrote phi_by_type.png")


# --------------------------------------------------------------------------- #
# Fig 3: RQ2 streamable fraction vs L, one line per delta, at theta=0.8        #
# --------------------------------------------------------------------------- #
def fig_rq2_surface(summary, theta=0.8):
    grid = summary["rq2_grid"]
    by_delta = defaultdict(list)
    for g in grid:
        if abs(g["theta"] - theta) < 1e-9 and g["frac"] is not None:
            by_delta[g["delta_wps"]].append((g["L_ms"], g["frac"]))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = ["#9bb4c4", NAVY, RUST]
    # collect all series first so we can de-duplicate labels across lines
    series = []
    for (d, pts), c in zip(sorted(by_delta.items()), colors):
        pts.sort()
        series.append((d, pts, c))
    annotated = set()  # (L, pct_str) already labelled
    for d, pts, c in series:
        xs = [p[0] for p in pts]
        ys = [100 * p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color=c, label=f"$\\delta$={d:g} w/s")
        for x, y in zip(xs, ys):
            key = (x, f"{y:.1f}")
            if key in annotated:
                continue
            annotated.add(key)
            ax.annotate(f"{y:.1f}%", xy=(x, y), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=7.5, color="black")
    ax.set_xlabel("tool latency $L$ (ms)")
    ax.set_ylabel(f"streamable %  ($\\theta$={theta})")
    ax.set_title("Fraction of queries that admit latency hiding")
    ax.set_ylim(0, 108)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGS}/rq2_surface.png", dpi=140)
    plt.close(fig)
    print("wrote rq2_surface.png")


# --------------------------------------------------------------------------- #
# Fig 4: RQ3 measured saving vs H-bound prediction (per question), colored by  #
# tool latency L. At L=600 the bound saturates (all H=600); at L=1000 it       #
# develops spread, yet measured savings still do not track H (rho approx 0).   #
# --------------------------------------------------------------------------- #
def _spearman(x, y):
    n = len(x)
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    import math
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx * sy else float("nan")


def fig_rq3_scatter(rows_by_L):
    series = [(L, rows) for L, rows in sorted(rows_by_L.items()) if rows]
    if not series:
        print("skip rq3_scatter (no rows)")
        return
    colors = {600.0: "#9bb4c4", 1000.0: RUST}
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    hi = 0.0
    for L, rows in series:
        meas = [fnum(r["measured_saved_ms"]) for r in rows]
        pred = [fnum(r["H_predicted_ms"]) for r in rows]
        hi = max(hi, max(meas), max(pred))
        rho = _spearman(pred, meas)
        ax.scatter(pred, meas, color=colors.get(L, NAVY), alpha=0.75,
                   edgecolor="white",
                   label=f"$L$={L:g} ms  ($\\rho$={rho:+.2f})")
    lo = min(0.0, min(fnum(r["measured_saved_ms"]) for _, rs in series for r in rs))
    ax.plot([0, hi * 1.1], [0, hi * 1.1], "--", color="grey", label="y = x (H bound)")
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel("$H$-bound predicted saving (ms)")
    ax.set_ylabel("measured saving (ms)")
    ax.set_ylim(lo * 1.15 if lo < 0 else 0, hi * 1.15)
    ax.set_title("RQ3: realized vs predicted saving (colored by $L$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/rq3_scatter.png", dpi=140)
    plt.close(fig)
    print("wrote rq3_scatter.png")


def main():
    rows_k3 = load_csv(f"{RESULTS}/stab_k3.csv")
    with open(f"{RESULTS}/stab_k3.summary.json") as f:
        summary_k3 = json.load(f)
    fig_phi_distribution(rows_k3)
    fig_phi_by_type(rows_k3)
    fig_rq2_surface(summary_k3)
    rows_by_L = {}
    for L, path in [(600.0, f"{RESULTS}/latency_k3.csv"),
                    (1000.0, f"{RESULTS}/latency_k3_L1000.csv")]:
        if os.path.exists(path):
            rows_by_L[L] = load_csv(path)
    if rows_by_L:
        fig_rq3_scatter(rows_by_L)


if __name__ == "__main__":
    main()
