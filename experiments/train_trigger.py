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
