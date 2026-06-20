# Component A: Classifier Trigger — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a per-word `P(t ≥ t_suf)` streaming-retrieval trigger, trained on CRAG split 1 and evaluated offline-analytically on split 0 against a fixed-interval baseline.

**Architecture:** Three new modules under `experiments/`. `trigger_features.py` runs the existing BM25 prefix sweep (via a `prefix_records()` refactor of `stabilization.py`) plus spaCy NER to emit a per-word feature/label CSV. `train_trigger.py` fits LogReg + GBDT, decodes a single fire point per question, scores an analytic latency/compute Pareto frontier vs. a swept fixed-interval baseline, runs a feature-group ablation, and writes a summary JSON + frontier plot. `audit_labels.py` samples retrieved-gold questions for a label-precision audit. All CPU-only.

**Tech Stack:** Python ≥3.10, stdlib, the vendored `streaming_rag.BM25`, and new optional extras: `scikit-learn`, `spacy` (+ `en_core_web_sm`), `numpy`; `pytest` for tests (dev extra).

## Global Constraints

- **Core stays zero-runtime-dep.** All new third-party imports live behind the `[trigger]` optional extra and are imported lazily inside functions, never at module top level (mirror `crag.html_to_text`'s guarded `from bs4 import ...`).
- **Train split 1, test split 0, no leakage.** `τ` and hyperparameters are selected only within split 1; split 0 is scored once.
- **Features are live-observable only:** `top1_stable_streak`, `top1_changed`, `t` (absolute word count, **never** `t/n`), `named_entity_detected`, `words_since_first_ne`, `question_word_type`.
- **Label** is `1[t ≥ t_suf]`, defined only on retrieved-gold questions; `label_sc = 1[t ≥ t_sc]` is always defined (secondary target).
- **Canonical operating point:** `L = 600` ms, `δ = 3` w/s. Premature-fire penalty `C_waste = L` (swept `{0.5L, L}`).
- **Reuse, don't duplicate:** retrieval goes through `prefix_records()`; latency uses `stabilization.hidden_latency_ms()`.
- **Module run context:** modules under `experiments/` import siblings by bare name (`from streaming_rag import BM25`). Tests put `experiments/` on `sys.path` via `pythonpath` in pyproject.
- **Artifacts:** write all outputs under `results/`.

---

### Task 1: Refactor `prefix_records()` + test harness setup

**Files:**
- Modify: `pyproject.toml` (add `[trigger]` and `[dev]` extras, pytest config)
- Modify: `experiments/stabilization.py` (extract `prefix_records`, keep `stabilization` behavior-identical)
- Create: `tests/conftest.py`
- Test: `tests/test_prefix_records.py`

**Interfaces:**
- Produces: `prefix_records(query: str, passages: list[str], top_k: int = 3, make_retriever=BM25) -> tuple[list[tuple[int, set[int]]], int]` — returns `(seq, n)` where `seq[t-1] = (top1_id, topk_id_set)` for prefix length `t` (1-based); `([], 0)` for empty query or passages.

- [ ] **Step 1: Add the `[trigger]` + `[dev]` extras and pytest config to `pyproject.toml`**

Add to `[project.optional-dependencies]` (after the `stats` entry):

```toml
# Component A classifier trigger (experiments/trigger_features.py, train_trigger.py):
# scikit-learn models, spaCy NER (run `python -m spacy download en_core_web_sm`), numpy.
trigger = ["scikit-learn>=1.3", "spacy>=3.7", "numpy>=1.24"]
# Test runner (no production code depends on it).
dev = ["pytest>=8.0"]
```

Append at end of file:

```toml
[tool.pytest.ini_options]
pythonpath = ["experiments"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `tests/conftest.py`**

```python
"""Make experiments/ importable and expose the fixture path."""
import os
import sys

HERE = os.path.dirname(__file__)
EXPERIMENTS = os.path.abspath(os.path.join(HERE, "..", "experiments"))
if EXPERIMENTS not in sys.path:
    sys.path.insert(0, EXPERIMENTS)
```

- [ ] **Step 3: Write the failing test**

`tests/test_prefix_records.py`:

```python
from stabilization import prefix_records, stabilization


def test_prefix_records_shape_and_monotonic_prefixes():
    passages = ["alpha beta", "gamma delta", "alpha gamma epsilon"]
    seq, n = prefix_records("alpha gamma epsilon", passages, top_k=2)
    assert n == 3
    assert len(seq) == 3
    for top1, ids in seq:
        assert isinstance(top1, int)
        assert isinstance(ids, set)
        assert len(ids) <= 2


def test_prefix_records_empty():
    assert prefix_records("", ["x"], top_k=3) == ([], 0)
    assert prefix_records("q", [], top_k=3) == ([], 0)


def test_stabilization_unchanged_after_refactor():
    passages = ["the eiffel tower is in paris", "paris is in france", "berlin germany"]
    s = stabilization("where is the eiffel tower", passages, gold={0}, top_k=2)
    assert s is not None
    assert s.n_words == 5
    assert s.t_suf is not None and 1 <= s.t_suf <= 5
    assert s.retrieved_gold is True
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_prefix_records.py -v`
Expected: FAIL with `ImportError: cannot import name 'prefix_records'`.

- [ ] **Step 5: Extract `prefix_records` in `experiments/stabilization.py`**

Replace the `_prefix_sequence` function (lines 33–44) and the head of `stabilization` (lines 52–55) so the sweep is a public function. New `prefix_records` (rename + early-empty guard) and `stabilization` calling it:

```python
def prefix_records(query: str, passages: list[str], top_k: int = 3,
                   make_retriever=BM25) -> tuple[list[tuple[int, set]], int]:
    """Retrieve over every prefix q[1:t]; return (seq, n) where
    seq[t-1] = (top1_id, set_of_topk_ids). ([], 0) if query or passages empty."""
    words = query.split()
    n = len(words)
    if not passages or n == 0:
        return [], 0
    retriever = make_retriever(passages)
    if hasattr(retriever, "prefix_topk"):
        return retriever.prefix_topk(words, top_k), n
    seq = []
    for t in range(1, n + 1):
        ranked = retriever.topk(" ".join(words[:t]), k=top_k)
        ids = [i for i, _ in ranked]
        seq.append((ids[0] if ids else -1, set(ids)))
    return seq, n
```

Then in `stabilization()` replace its body's first lines:

```python
def stabilization(query: str, passages: list[str], gold: set[int], top_k: int = 3,
                  make_retriever=BM25) -> Optional[Stab]:
    seq, n = prefix_records(query, passages, top_k, make_retriever)
    if not seq:
        return None
    full_top1 = seq[-1][0]
    # ... (rest of the function unchanged: t_sc, volatility, t_suf, return Stab)
```

Delete the old `_prefix_sequence` definition and its call. Leave all metric logic (lines 57–93) byte-identical.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_prefix_records.py -v`
Expected: 3 passed.

- [ ] **Step 7: Smoke-test the unchanged pipeline**

Run: `uv run experiments/run_study.py --data data/crag_fixture.jsonl.bz2 --split 0 --out /tmp/stab_smoke.csv`
Expected: exits 0, prints the stabilization summary (proves the refactor didn't change behavior).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml experiments/stabilization.py tests/conftest.py tests/test_prefix_records.py
git commit -m "refactor: extract prefix_records() for trigger feature reuse; add pytest harness"
```

---

### Task 2: Query-side feature helpers (question word, entity offsets)

**Files:**
- Create: `experiments/trigger_features.py`
- Test: `tests/test_trigger_features.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `question_word_type(query: str) -> str` — one of `who/what/when/where/which/why/how/other` (first interrogative in the first 3 words, lowercased; else `other`).
  - `char_spans_to_word_offsets(query: str, char_spans: list[tuple[int, int]]) -> list[int]` — for each entity char-start, the 1-based word index it falls in; sorted, de-duplicated.
  - `first_ne_position(offsets: list[int]) -> int | None` — `min(offsets)` or `None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_trigger_features.py`:

```python
from trigger_features import (
    question_word_type, char_spans_to_word_offsets, first_ne_position,
)


def test_question_word_type():
    assert question_word_type("who makes the playstation") == "who"
    assert question_word_type("Where is the Eiffel Tower") == "where"
    assert question_word_type("name the tallest mountain") == "other"
    assert question_word_type("") == "other"


def test_char_spans_to_word_offsets():
    q = "who makes the playstation console"   # word offsets: who=1 makes=2 the=3 playstation=4 console=5
    # "playstation" starts at char index 15
    assert q.index("playstation") == 15
    assert char_spans_to_word_offsets(q, [(15, 26)]) == [4]
    # two entities, unsorted input, dedup
    spans = [(q.index("console"), q.index("console") + 7), (15, 26)]
    assert char_spans_to_word_offsets(q, spans) == [4, 5]


def test_first_ne_position():
    assert first_ne_position([4, 5]) == 4
    assert first_ne_position([]) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_trigger_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trigger_features'`.

- [ ] **Step 3: Create `experiments/trigger_features.py` with the helpers**

```python
"""Per-word feature/label extraction for the Component A classifier trigger.

Reuses stabilization.prefix_records for the retrieval-side signal and spaCy NER
(lazy import, [trigger] extra) for the entity-side signal. See
docs/superpowers/specs/2026-06-20-component-a-trigger-design.md.
"""
from __future__ import annotations

from typing import Optional

_QWORDS = ("who", "what", "when", "where", "which", "why", "how")


def question_word_type(query: str) -> str:
    for w in query.lower().split()[:3]:
        if w in _QWORDS:
            return w
    return "other"


def char_spans_to_word_offsets(query: str, char_spans: list[tuple[int, int]]) -> list[int]:
    """Map each entity char-start to the 1-based index of the word it lands in."""
    # Build (char_start, word_index) for each whitespace-split word.
    bounds = []
    pos = 0
    for idx, word in enumerate(query.split(), start=1):
        start = query.index(word, pos)
        bounds.append((start, start + len(word), idx))
        pos = start + len(word)
    offsets = set()
    for cstart, _cend in char_spans:
        for wstart, wend, widx in bounds:
            if wstart <= cstart < wend:
                offsets.add(widx)
                break
    return sorted(offsets)


def first_ne_position(offsets: list[int]) -> Optional[int]:
    return min(offsets) if offsets else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_trigger_features.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/trigger_features.py tests/test_trigger_features.py
git commit -m "feat: question-word and entity-offset feature helpers"
```

---

### Task 3: spaCy NER wrapper

**Files:**
- Modify: `experiments/trigger_features.py`
- Test: `tests/test_trigger_features.py`

**Interfaces:**
- Consumes: `char_spans_to_word_offsets` (Task 2).
- Produces: `spacy_ner(query: str) -> list[int]` — 1-based word offsets of entity starts, via a cached `en_core_web_sm` pipeline; raises a clear error if spaCy/model is missing.

- [ ] **Step 1: Write the failing test (logic via a stub, no model download)**

Add to `tests/test_trigger_features.py`:

```python
import trigger_features as tf


def test_spacy_ner_uses_char_span_mapping(monkeypatch):
    # Stub the spaCy doc.ents so the test needs no model download.
    class _Ent:
        def __init__(self, start_char): self.start_char = start_char

    class _Doc:
        def __init__(self, ents): self.ents = ents

    q = "who founded microsoft corporation"
    def fake_nlp(text):
        return _Doc([_Ent(text.index("microsoft"))])
    monkeypatch.setattr(tf, "_load_nlp", lambda: fake_nlp)

    assert tf.spacy_ner(q) == [3]  # "microsoft" is the 3rd word
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_trigger_features.py::test_spacy_ner_uses_char_span_mapping -v`
Expected: FAIL with `AttributeError: module 'trigger_features' has no attribute '_load_nlp'`.

- [ ] **Step 3: Add the lazy spaCy wrapper to `experiments/trigger_features.py`**

```python
_NLP = None


def _load_nlp():
    """Lazily load and cache the spaCy pipeline ([trigger] extra)."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "spaCy not installed. Install with: uv sync --extra trigger"
            ) from e
        try:
            _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "parser"])
        except OSError as e:
            raise OSError(
                "spaCy model missing. Run: python -m spacy download en_core_web_sm"
            ) from e
    return _NLP


def spacy_ner(query: str) -> list[int]:
    """1-based word offsets of named-entity starts in the full query."""
    if not query.split():
        return []
    doc = _load_nlp()(query)
    spans = [(ent.start_char, ent.start_char) for ent in doc.ents]
    return char_spans_to_word_offsets(query, spans)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_trigger_features.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/trigger_features.py tests/test_trigger_features.py
git commit -m "feat: lazy spaCy NER wrapper returning entity word offsets"
```

---

### Task 4: Per-word row builder (features + labels)

**Files:**
- Modify: `experiments/trigger_features.py`
- Test: `tests/test_trigger_features.py`

**Interfaces:**
- Consumes: `question_word_type`, `first_ne_position` (Task 2).
- Produces: `per_word_rows(meta: dict, seq: list[tuple[int, set]], n: int, t_suf: Optional[int], t_sc: int, ne_offsets: list[int]) -> list[dict]` — one dict per word position `t = 1..n`. `meta` carries `interaction_id, question_type, domain, query`. Row keys: `interaction_id, question_type, domain, retrieved_gold, n_words, t, t_suf, t_sc, top1_stable_streak, top1_changed, named_entity_detected, words_since_first_ne, question_word_type, label, label_sc`. `label` is `""` when `t_suf` is `None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_trigger_features.py`:

```python
from trigger_features import per_word_rows


def test_per_word_rows_features_and_labels():
    # seq: top1 ids by prefix length 1..4; stabilizes on id 7 from t=3.
    seq = [(2, {2}), (5, {5, 2}), (7, {7, 5}), (7, {7, 9})]
    meta = {"interaction_id": "q1", "question_type": "simple",
            "domain": "music", "query": "who made the playstation"}
    rows = per_word_rows(meta, seq, n=4, t_suf=3, t_sc=3, ne_offsets=[4])
    assert len(rows) == 4
    # streak resets on change, grows while stable
    assert [r["top1_stable_streak"] for r in rows] == [1, 1, 1, 2]
    assert [r["top1_changed"] for r in rows] == [0, 1, 1, 0]
    # label = 1[t >= t_suf]; t_suf=3
    assert [r["label"] for r in rows] == [0, 0, 1, 1]
    assert [r["label_sc"] for r in rows] == [0, 0, 1, 1]
    # entity first appears at word 4 -> detected only at t=4
    assert [r["named_entity_detected"] for r in rows] == [0, 0, 0, 1]
    assert [r["words_since_first_ne"] for r in rows] == [0, 0, 0, 0]
    assert rows[0]["question_word_type"] == "who"
    assert rows[0]["retrieved_gold"] is True


def test_per_word_rows_ungroundable_blank_label():
    seq = [(2, {2}), (2, {2})]
    meta = {"interaction_id": "q2", "question_type": "false_premise",
            "domain": "x", "query": "is the moon cheese"}
    rows = per_word_rows(meta, seq, n=2, t_suf=None, t_sc=1, ne_offsets=[])
    assert [r["label"] for r in rows] == ["", ""]
    assert [r["label_sc"] for r in rows] == [1, 1]
    assert rows[0]["retrieved_gold"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_trigger_features.py::test_per_word_rows_features_and_labels -v`
Expected: FAIL with `ImportError: cannot import name 'per_word_rows'`.

- [ ] **Step 3: Add `per_word_rows` to `experiments/trigger_features.py`**

```python
def per_word_rows(meta: dict, seq: list[tuple[int, set]], n: int,
                  t_suf: Optional[int], t_sc: int, ne_offsets: list[int]) -> list[dict]:
    qword = question_word_type(meta["query"])
    first_ne = first_ne_position(ne_offsets)
    rows = []
    streak = 1
    for i in range(n):
        t = i + 1
        top1 = seq[i][0]
        changed = 1 if (i > 0 and seq[i - 1][0] != top1) else 0
        streak = 1 if changed else (streak + 1 if i > 0 else 1)
        ne_det = 1 if (first_ne is not None and t >= first_ne) else 0
        since = max(0, t - first_ne) if (first_ne is not None and t >= first_ne) else 0
        rows.append({
            "interaction_id": meta["interaction_id"],
            "question_type": meta["question_type"],
            "domain": meta["domain"],
            "retrieved_gold": t_suf is not None,
            "n_words": n,
            "t": t,
            "t_suf": t_suf if t_suf is not None else "",
            "t_sc": t_sc,
            "top1_stable_streak": streak,
            "top1_changed": changed,
            "named_entity_detected": ne_det,
            "words_since_first_ne": since,
            "question_word_type": qword,
            "label": (1 if t >= t_suf else 0) if t_suf is not None else "",
            "label_sc": 1 if t >= t_sc else 0,
        })
    return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_trigger_features.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/trigger_features.py tests/test_trigger_features.py
git commit -m "feat: per-word feature/label row builder"
```

---

### Task 5: `trigger_features.py` CLI driver (writes the per-word CSV)

**Files:**
- Modify: `experiments/trigger_features.py`
- Test: `tests/test_trigger_features.py`

**Interfaces:**
- Consumes: `crag.load_crag`, `stabilization.prefix_records` + `stabilization.stabilization`, `per_word_rows`, `spacy_ner`.
- Produces: `extract(data, split, top_k, ner_fn) -> list[dict]` (all per-word rows for a split) and a `main()` CLI writing `--out`.

- [ ] **Step 1: Write the failing test (fixture end-to-end, NER injected)**

Add to `tests/test_trigger_features.py`:

```python
from trigger_features import extract, FEATURE_FIELDS


def test_extract_on_fixture_schema_and_counts():
    rows = extract("data/crag_fixture.jsonl.bz2", split=0, top_k=3,
                   ner_fn=lambda q: [])   # no-NER stub keeps it fast/deterministic
    assert rows, "fixture produced no rows"
    # one row per word position per question
    by_q = {}
    for r in rows:
        by_q.setdefault(r["interaction_id"], []).append(r)
    for q, qr in by_q.items():
        assert [r["t"] for r in qr] == list(range(1, qr[0]["n_words"] + 1))
    # every declared field is present
    for r in rows:
        assert set(FEATURE_FIELDS).issubset(r.keys())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_trigger_features.py::test_extract_on_fixture_schema_and_counts -v`
Expected: FAIL with `ImportError: cannot import name 'extract'`.

- [ ] **Step 3: Add `extract`, `FEATURE_FIELDS`, and `main` to `experiments/trigger_features.py`**

```python
import argparse
import csv

from crag import load_crag
from stabilization import prefix_records, stabilization

FEATURE_FIELDS = [
    "interaction_id", "question_type", "domain", "retrieved_gold", "n_words", "t",
    "t_suf", "t_sc", "top1_stable_streak", "top1_changed",
    "named_entity_detected", "words_since_first_ne", "question_word_type",
    "label", "label_sc",
]


def extract(data: str, split: int, top_k: int, ner_fn=spacy_ner,
            limit: Optional[int] = None) -> list[dict]:
    out = []
    for ex in load_crag(data, split=split, limit=limit):
        seq, n = prefix_records(ex.query, ex.passages, top_k)
        if not seq:
            continue
        s = stabilization(ex.query, ex.passages, ex.gold, top_k=top_k)
        if s is None:
            continue
        ne_offsets = ner_fn(ex.query)
        meta = {"interaction_id": ex.interaction_id, "question_type": ex.question_type,
                "domain": ex.domain, "query": ex.query}
        out.extend(per_word_rows(meta, seq, n, s.t_suf, s.t_sc, ne_offsets))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = extract(args.data, args.split, args.top_k, limit=args.limit)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} per-word rows -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_trigger_features.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/trigger_features.py tests/test_trigger_features.py
git commit -m "feat: trigger_features CLI emits per-word feature/label CSV"
```

- [ ] **Step 6: Generate the real feature tables (the labels that don't exist yet)**

Run (split 1 = train; split 0 = test; both need the spaCy model installed first):

```bash
uv sync --extra trigger --extra dev
uv run python -m spacy download en_core_web_sm
uv run experiments/trigger_features.py --data data/crag_task_1_and_2_dev_v4.jsonl.bz2 --split 1 --top-k 3 --out results/trigger_features.split1.csv
uv run experiments/trigger_features.py --data data/crag_task_1_and_2_dev_v4.jsonl.bz2 --split 0 --top-k 3 --out results/trigger_features.split0.csv
```

Expected: two CSVs, each "wrote N per-word rows". Commit them:

```bash
git add results/trigger_features.split0.csv results/trigger_features.split1.csv
git commit -m "data: per-word trigger feature tables (split 0 test, split 1 train)"
```

---

### Task 6: Model fit, fire decoding, analytic saving

**Files:**
- Create: `experiments/train_trigger.py`
- Test: `tests/test_train_trigger.py`

**Interfaces:**
- Produces:
  - `FEATURES: list[str]` (numeric feature names) and `QWORD_LEVELS: list[str]`.
  - `load_features(path: str) -> list[dict]` — DictReader with numeric casting.
  - `to_matrix(rows: list[dict]) -> tuple[list[list[float]], list[int]]` — `(X, y_suf)` for retrieved-gold rows (one-hots `question_word_type`).
  - `fit_models(X, y) -> dict` — `{"logreg": Pipeline, "gbdt": GradientBoostingClassifier}`.
  - `decode_fire_t(ts: list[int], probs: list[float], tau: float) -> int | None` — smallest `t` with `prob ≥ tau`.
  - `analytic_saving(fire_t, t_suf, n, L, delta, c_waste) -> float`.

- [ ] **Step 1: Write the failing test**

`tests/test_train_trigger.py`:

```python
from train_trigger import decode_fire_t, analytic_saving


def test_decode_fire_t():
    assert decode_fire_t([1, 2, 3], [0.1, 0.6, 0.9], tau=0.5) == 2
    assert decode_fire_t([1, 2, 3], [0.1, 0.2, 0.3], tau=0.5) is None


def test_analytic_saving_correct_fire():
    # correct fire at t_suf: H = min(L, (n - fire_t)/delta * 1000)
    # n=10, fire_t=4, delta=3 -> residual = 6/3*1000 = 2000ms, capped at L=600
    assert analytic_saving(4, t_suf=4, n=10, L=600, delta=3, c_waste=600) == 600.0


def test_analytic_saving_premature_is_penalized():
    # fire_t=2 < t_suf=5: saving = H(t_suf) - c_waste
    # H(t_suf=5): residual = (10-5)/3*1000 = 1666 -> cap 600; minus c_waste 600 = 0
    assert analytic_saving(2, t_suf=5, n=10, L=600, delta=3, c_waste=600) == 0.0
    # smaller penalty -> positive
    assert analytic_saving(2, t_suf=5, n=10, L=600, delta=3, c_waste=300) == 300.0


def test_analytic_saving_never_fires():
    assert analytic_saving(None, t_suf=5, n=10, L=600, delta=3, c_waste=600) == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_train_trigger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_trigger'`.

- [ ] **Step 3: Create `experiments/train_trigger.py` with the pure functions + model fit**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_train_trigger.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/train_trigger.py tests/test_train_trigger.py
git commit -m "feat: trigger model fit, fire decoding, analytic saving"
```

---

### Task 7: Evaluation — baseline, frontier, ablation, summary, plot, label audit

**Files:**
- Modify: `experiments/train_trigger.py`
- Create: `experiments/audit_labels.py`
- Test: `tests/test_train_trigger.py`, `tests/test_audit_labels.py`

**Interfaces:**
- Consumes: everything in Task 6.
- Produces:
  - `fixed_interval_eval(t_suf: int, n: int, interval: int) -> tuple[int | None, int]` — `(fire_t, n_calls)` for the baseline.
  - `group_by_question(rows: list[dict]) -> dict[str, list[dict]]`.
  - `spearman(a: list[float], b: list[float]) -> float`.
  - `evaluate(questions, model, tau, target, L, delta, c_waste) -> dict` — metric block.
  - `audit_labels.sample_audit(rows: list[dict], n: int, seed: int) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_train_trigger.py`:

```python
from train_trigger import fixed_interval_eval, spearman


def test_fixed_interval_eval():
    # interval=2: fires at 2,4,6...; first >= t_suf=5 is 6 -> 3 calls
    assert fixed_interval_eval(t_suf=5, n=10, interval=2) == (6, 3)
    # never reaches gold within n -> None, floor(n/interval) calls
    assert fixed_interval_eval(t_suf=9, n=7, interval=2) == (None, 3)


def test_spearman_monotonic():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
```

Create `tests/test_audit_labels.py`:

```python
from audit_labels import sample_audit


def _rows(n):
    return [{"interaction_id": f"q{i}", "query": f"query {i}",
             "retrieved_gold": True, "t_suf": 2, "gold_passage": f"p{i}"} for i in range(n)]


def test_sample_audit_deterministic_and_sized():
    rows = _rows(50)
    a = sample_audit(rows, n=10, seed=0)
    b = sample_audit(rows, n=10, seed=0)
    assert len(a) == 10
    assert [r["interaction_id"] for r in a] == [r["interaction_id"] for r in b]
    assert all(r["is_answer_bearing"] == "" for r in a)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_train_trigger.py::test_fixed_interval_eval tests/test_audit_labels.py -v`
Expected: FAIL (`cannot import name 'fixed_interval_eval'`, `No module named 'audit_labels'`).

- [ ] **Step 3: Add evaluation functions to `experiments/train_trigger.py`**

```python
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
    return cov / (va * vb) if va and vb else float("nan")


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
```

- [ ] **Step 4: Add the frontier/ablation/summary `main` to `experiments/train_trigger.py`**

```python
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
        # rebuild matrices with the reduced numeric feature set
        def vec(r):
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
        ax.set_xlabel("retrieval calls / question"); ax.set_ylabel("mean analytic saving (ms)")
        ax.legend(); fig.savefig(args.plot_out, dpi=120, bbox_inches="tight")
        print(f"wrote {args.plot_out}")
    except ImportError:
        print("matplotlib not installed; skipped plot (install --extra plot)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create `experiments/audit_labels.py`**

```python
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
        w.writeheader(); w.writerows(audit)
    print(f"wrote {len(audit)} audit rows -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run all tests to verify they pass**

Run: `uv run --extra dev pytest tests/ -v`
Expected: all pass (Task 1–7 tests green).

- [ ] **Step 7: Commit**

```bash
git add experiments/train_trigger.py experiments/audit_labels.py tests/test_train_trigger.py tests/test_audit_labels.py
git commit -m "feat: trigger evaluation (frontier, baseline, ablation, summary, plot) + label audit"
```

- [ ] **Step 8: Run the real evaluation end-to-end**

Run (requires the feature tables from Task 5, Step 6):

```bash
uv run --extra trigger --extra plot experiments/train_trigger.py \
  --train results/trigger_features.split1.csv \
  --test results/trigger_features.split0.csv \
  --summary-json results/trigger.summary.json --plot-out results/trigger_frontier.png
uv run --extra trigger experiments/audit_labels.py \
  --data data/crag_task_1_and_2_dev_v4.jsonl.bz2 --split 0 --n 100 \
  --out results/label_audit.csv
```

Expected: `results/trigger.summary.json`, `results/trigger_frontier.png`, `results/label_audit.csv`. Sanity-check the summary: `operating_point.logreg.mean_saving_ms`, the `frontier` arrays, `ablation.retrieval_stability` vs `ablation.entity` Spearman, and `importances`.

```bash
git add results/trigger.summary.json results/trigger_frontier.png results/label_audit.csv
git commit -m "results: Component A trigger frontier, ablation, importances, label-audit sample"
```

---

## Self-Review

**Spec coverage (every spec section maps to a task):**
- §2 `prefix_records` refactor → Task 1. §3 modules/extras/artifacts → Tasks 1,5,6,7. §4 features (incl. NER fidelity) → Tasks 2,3,4. §5 labels + t_sc secondary → Task 4 (`label_sc`), Task 7 (`sc_safety_point`). §6 analytic saving + frontier + swept baseline → Tasks 6,7. §6 populations → `evaluate(target=...)` Task 7. §6 ablation/importances → Task 7. §7 label audit → Task 7 (`audit_labels.py`). §8 train/test discipline → Task 5 Step 6 (split 1 train / split 0 test). §9 deferral — nothing to build. §10 kill-criteria — reporting, surfaced by the summary keys.
- **Gap noted & accepted:** within-split-1 cross-validation for τ/hyperparameter selection (§8) is represented by the τ-sweep `frontier` over split 0; the plan reports the full frontier rather than auto-selecting τ on split 1. Tighten to an in-split-1 CV selection if a single headline τ is needed for the paper — flagged here, not silently dropped.

**Placeholder scan:** none — every code/test step carries complete code.

**Type consistency:** `prefix_records` returns `(seq, n)` consumed identically in Tasks 1/5. `_feat_vector` order (`FEATURES` + `QWORD_LEVELS`) matches `importances` zip in Task 7. `decode_fire_t`, `analytic_saving`, `fixed_interval_eval` signatures identical across definition (Task 6/7) and tests. `per_word_rows` row keys are the superset of `FEATURE_FIELDS` (Task 5) plus `label_sc`; `FEATURE_FIELDS` enumerates the CSV columns and `load_features` casts exactly those.
