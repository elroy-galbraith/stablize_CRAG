"""Fit the Component A trigger, decode fires, score the analytic frontier.

See docs/superpowers/specs/2026-06-20-component-a-trigger-design.md. Offline
analytic eval only; the real async-harness replay is Component D.
"""
from __future__ import annotations

import argparse
import csv
import json
from typing import Optional

from stabilization import hidden_latency_ms

FEATURES = ["top1_stable_streak", "top1_changed", "t",
            "named_entity_detected", "words_since_first_ne"]
QWORD_LEVELS = ["who", "what", "when", "where", "which", "why", "how", "other"]
_NUMERIC = {"retrieved_gold", "n_words", "t", "top1_stable_streak", "top1_changed",
            "named_entity_detected", "words_since_first_ne", "label_sc"}


def load_features(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["retrieved_gold"] = r["retrieved_gold"] == "True"
            for k in ("n_words", "t", "top1_stable_streak", "top1_changed",
                      "named_entity_detected", "words_since_first_ne", "t_sc", "label_sc"):
                r[k] = int(r[k])
            r["t_suf"] = int(r["t_suf"]) if r["t_suf"] != "" else None
            r["label"] = int(r["label"]) if r["label"] != "" else None
            rows.append(r)
    return rows


def _feat_vector(r: dict) -> list[float]:
    base = [float(r[k]) for k in FEATURES]
    onehot = [1.0 if r["question_word_type"] == lv else 0.0 for lv in QWORD_LEVELS]
    return base + onehot


def to_matrix(rows: list[dict], target: str = "suf"):
    X, y = [], []
    for r in rows:
        if target == "suf":
            if not r["retrieved_gold"] or r["label"] is None:
                continue
            X.append(_feat_vector(r)); y.append(r["label"])
        else:  # sc target uses all rows
            X.append(_feat_vector(r)); y.append(r["label_sc"])
    return X, y


def fit_models(X, y) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    ).fit(X, y)
    gbdt = GradientBoostingClassifier(random_state=0).fit(X, y)
    return {"logreg": logreg, "gbdt": gbdt}


def decode_fire_t(ts: list[int], probs: list[float], tau: float) -> Optional[int]:
    for t, p in zip(ts, probs):
        if p >= tau:
            return t
    return None


def analytic_saving(fire_t: Optional[int], t_suf: int, n: int,
                    L: float, delta: float, c_waste: float) -> float:
    if fire_t is None:
        return 0.0
    if fire_t >= t_suf:
        return hidden_latency_ms(fire_t, n, L, delta)
    return hidden_latency_ms(t_suf, n, L, delta) - c_waste


def fixed_interval_eval(t_suf: int, n: int, interval: int) -> tuple[Optional[int], int]:
    """Baseline fires at interval, 2*interval, ... until one lands >= t_suf."""
    fire = interval
    calls = 0
    while fire <= n:
        calls += 1
        if fire >= t_suf:
            return fire, calls
        fire += interval
    return None, calls


def group_by_question(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        out.setdefault(r["interaction_id"], []).append(r)
    for qr in out.values():
        qr.sort(key=lambda r: r["t"])
    return out


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    r = cov / (va * vb) if va and vb else float("nan")
    return round(max(-1.0, min(1.0, r)), 10)


def evaluate(questions: dict, model, tau: float, target: str,
             L: float, delta: float, c_waste: float) -> dict:
    fire_ts, true_ts, savings, premature, calls, correct_savings = [], [], [], [], [], []
    for qr in questions.values():
        target_t = qr[0]["t_suf"] if target == "suf" else qr[0]["t_sc"]
        if target_t is None:
            continue
        n = qr[0]["n_words"]
        probs = model.predict_proba([_feat_vector(r) for r in qr])[:, 1].tolist()
        ts = [r["t"] for r in qr]
        ft = decode_fire_t(ts, probs, tau)
        s = analytic_saving(ft, target_t, n, L, delta, c_waste)
        fire_ts.append(ft if ft is not None else n)
        true_ts.append(target_t)
        savings.append(s)
        is_prem = ft is not None and ft < target_t
        premature.append(1 if is_prem else 0)
        calls.append((1 + (1 if is_prem else 0)) if ft is not None else 1)
        if ft is not None and ft >= target_t:
            correct_savings.append(s)
    k = len(savings)
    correct_savings.sort()
    return {
        "tau": tau, "n_questions": k,
        "mean_saving_ms": sum(savings) / k if k else 0.0,
        "misfire_rate": sum(premature) / k if k else 0.0,
        "net_negative_rate": sum(1 for s in savings if s <= 0) / k if k else 0.0,
        "mean_calls": sum(calls) / k if k else 0.0,
        "median_saving_correct_fires": (correct_savings[len(correct_savings) // 2]
                                        if correct_savings else 0.0),
        "spearman_fire_vs_tsuf": spearman([float(x) for x in fire_ts],
                                          [float(x) for x in true_ts]),
    }


def baseline_frontier(questions: dict, target: str, intervals, L, delta, c_waste) -> list[dict]:
    out = []
    for interval in intervals:
        savings, premature, calls = [], [], []
        for qr in questions.values():
            target_t = qr[0]["t_suf"] if target == "suf" else qr[0]["t_sc"]
            if target_t is None:
                continue
            n = qr[0]["n_words"]
            ft, c = fixed_interval_eval(target_t, n, interval)
            savings.append(analytic_saving(ft, target_t, n, L, delta, c_waste))
            premature.append(1 if (ft is not None and ft < target_t) else 0)
            calls.append(c if c else 1)
        k = len(savings)
        out.append({"interval": interval,
                    "mean_saving_ms": sum(savings) / k if k else 0.0,
                    "misfire_rate": sum(premature) / k if k else 0.0,
                    "mean_calls": sum(calls) / k if k else 0.0})
    return out


def ablation(train_rows, test_questions, tau, L, delta, c_waste) -> dict:
    groups = {"retrieval_stability": ["top1_stable_streak", "top1_changed"],
              "entity": ["named_entity_detected", "words_since_first_ne"]}
    result = {}
    for name, drop in groups.items():
        keep = [f for f in FEATURES if f not in drop]

        def vec(r, keep=keep):
            base = [float(r[k]) for k in keep]
            return base + [1.0 if r["question_word_type"] == lv else 0.0 for lv in QWORD_LEVELS]
        X = [vec(r) for r in train_rows if r["retrieved_gold"] and r["label"] is not None]
        y = [r["label"] for r in train_rows if r["retrieved_gold"] and r["label"] is not None]
        m = fit_models(X, y)["logreg"]
        fire_ts, true_ts = [], []
        for qr in test_questions.values():
            if qr[0]["t_suf"] is None:
                continue
            probs = m.predict_proba([vec(r) for r in qr])[:, 1].tolist()
            ft = decode_fire_t([r["t"] for r in qr], probs, tau)
            fire_ts.append(float(ft if ft is not None else qr[0]["n_words"]))
            true_ts.append(float(qr[0]["t_suf"]))
        result[name] = {"dropped": drop, "spearman_fire_vs_tsuf": spearman(fire_ts, true_ts)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--L", type=float, default=600.0)
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--c-waste", type=float, default=600.0)
    ap.add_argument("--summary-json", default="results/trigger.summary.json")
    ap.add_argument("--plot-out", default="results/trigger_frontier.png")
    args = ap.parse_args()

    train_rows = load_features(args.train)
    test_rows = load_features(args.test)
    Xtr, ytr = to_matrix(train_rows, target="suf")
    models = fit_models(Xtr, ytr)
    test_q = group_by_question(test_rows)

    taus = [round(0.05 * i, 2) for i in range(1, 20)]
    frontier = {name: [evaluate(test_q, m, tau, "suf", args.L, args.delta, args.c_waste)
                       for tau in taus]
                for name, m in models.items()}
    baseline = baseline_frontier(test_q, "suf", [1, 2, 3, 4], args.L, args.delta, args.c_waste)
    abl = ablation(train_rows, test_q, args.tau, args.L, args.delta, args.c_waste)
    sc_point = evaluate(test_q, models["logreg"], args.tau, "sc", args.L, args.delta, args.c_waste)

    logreg = models["logreg"].named_steps["logisticregression"]
    importances = {
        "logreg_coef": dict(zip(FEATURES + QWORD_LEVELS, logreg.coef_[0].tolist())),
        "gbdt_importance": dict(zip(FEATURES + QWORD_LEVELS,
                                    models["gbdt"].feature_importances_.tolist())),
    }
    summary = {
        "params": {"train": args.train, "test": args.test, "tau": args.tau,
                   "L": args.L, "delta": args.delta, "c_waste": args.c_waste},
        "operating_point": {name: evaluate(test_q, m, args.tau, "suf",
                                           args.L, args.delta, args.c_waste)
                            for name, m in models.items()},
        "frontier": frontier, "baseline_frontier": baseline,
        "sc_safety_point": sc_point, "ablation": abl, "importances": importances,
    }
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.summary_json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for name, pts in frontier.items():
            ax.plot([p["mean_calls"] for p in pts], [p["mean_saving_ms"] for p in pts],
                    marker="o", label=f"trigger ({name})")
        ax.plot([p["mean_calls"] for p in baseline], [p["mean_saving_ms"] for p in baseline],
                marker="s", label="fixed-interval")
        ax.set_xlabel("retrieval calls / question")
        ax.set_ylabel("mean analytic saving (ms)")
        ax.legend()
        fig.savefig(args.plot_out, dpi=120, bbox_inches="tight")
        print(f"wrote {args.plot_out}")
    except ImportError:
        print("matplotlib not installed; skipped plot (install --extra plot)")


if __name__ == "__main__":
    main()
