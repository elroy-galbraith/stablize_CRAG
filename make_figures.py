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

RESULTS = "results"
FIGS = "paper/figures"
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
# Fig 2: phi_suf by question type (k=3), sorted, with counts                   #
# --------------------------------------------------------------------------- #
def fig_phi_by_type(summary):
    rq4 = summary["rq4_phi_suf_by_question_type"]
    items = sorted(rq4.items(), key=lambda kv: kv[1]["mean_phi_suf"])
    labels = [k.replace("_", "\n") for k, _ in items]
    means = [v["mean_phi_suf"] for _, v in items]
    counts = [v["n"] for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(labels, means, color=NAVY, edgecolor="white")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"n={c}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(r"mean $\phi_{suf}$")
    ax.set_title("Sufficiency stabilization by question type (earlier $\\to$ later)")
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
    for (d, pts), c in zip(sorted(by_delta.items()), colors):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [100 * p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color=c, label=f"$\\delta$={d:g} w/s")
    ax.set_xlabel("tool latency $L$ (ms)")
    ax.set_ylabel(f"streamable %  ($\\theta$={theta})")
    ax.set_title("Fraction of queries that admit latency hiding")
    ax.set_ylim(0, 102)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGS}/rq2_surface.png", dpi=140)
    plt.close(fig)
    print("wrote rq2_surface.png")


# --------------------------------------------------------------------------- #
# Fig 4: RQ3 measured saving vs H-bound prediction (per question)             #
# --------------------------------------------------------------------------- #
def fig_rq3_scatter(rows):
    if not rows:
        print("skip rq3_scatter (no rows)")
        return
    meas = [fnum(r["measured_saved_ms"]) for r in rows]
    pred = [fnum(r["H_predicted_ms"]) for r in rows]
    hi = max(max(meas), max(pred)) * 1.1
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(pred, meas, color=NAVY, alpha=0.7, edgecolor="white")
    ax.plot([0, hi], [0, hi], "--", color="grey", label="y = x (H bound)")
    ax.set_xlabel("$H$-bound predicted saving (ms)")
    ax.set_ylabel("measured saving (ms)")
    ax.set_title("RQ3: realized vs predicted latency saving")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGS}/rq3_scatter.png", dpi=140)
    plt.close(fig)
    print("wrote rq3_scatter.png")


def main():
    rows_k3 = load_csv(f"{RESULTS}/stab_k3.csv")
    with open(f"{RESULTS}/stab_k3.summary.json") as f:
        summary_k3 = json.load(f)
    fig_phi_distribution(rows_k3)
    fig_phi_by_type(summary_k3)
    fig_rq2_surface(summary_k3)
    lat_path = f"{RESULTS}/latency_k3.csv"
    if os.path.exists(lat_path):
        fig_rq3_scatter(load_csv(lat_path))


if __name__ == "__main__":
    main()
