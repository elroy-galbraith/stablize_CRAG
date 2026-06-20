# Second-Benchmark Generalization (HotpotQA + shared infra) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the CRAG stabilization + trigger pipeline on HotpotQA (clean gold passages) to test whether early-stabilization generalizes, whether the over-grounding artifact replicates, and whether the trigger's per-query signal grows where t_suf spreads — reusing the existing pipeline through one `CragExample` contract.

**Architecture:** A thin HotpotQA loader emits the existing `CragExample` shape, so `stabilization`, `trigger_features`, `train_trigger`, and `grounding_precision` run unchanged. A small `benchmark_stats.py` computes φ_suf under clean gold vs CRAG-style string-grounding on the same questions (the bias arm). A single-fire baseline is added to `train_trigger` for an equal-compute trigger comparison.

**Tech Stack:** Python ≥3.10, stdlib, vendored `streaming_rag.BM25`, the existing `[trigger]` extra (scikit-learn, spaCy), and a new `[bench]` extra (`datasets`, HuggingFace).

## Global Constraints

- **Core stays zero-runtime-dependency.** `datasets` is imported lazily inside the loader, never at module top level (mirror the existing lazy `import spacy`/`from bs4`).
- **One contract:** every loader yields `crag.CragExample` (`interaction_id, query, answer, alt_ans, domain, question_type, static_or_dynamic, split, passages, gold`). Nothing downstream is modified to accommodate a benchmark.
- **HotpotQA passage mapping:** 1 paragraph = 1 passage (no `chunk_text`), so passage index == context-paragraph index and `gold` = indices of supporting-fact paragraphs. Gold is the **shipped** label, never string-matched.
- **HotpotQA HF source:** repo `hotpotqa/hotpot_qa`, config `distractor`; `split=0`→HF `validation` (test), `split=1`→HF `train`. No `trust_remote_code`.
- **Train split 1 / test split 0, no leakage** (same discipline as CRAG Component A).
- **Two gold definitions per question:** clean (shipped) populates `.gold`; string-grounded is computed on demand via `crag.gold_passage_ids(answer, alt_ans, passages)`.
- **Artifacts** under `results/hotpotqa/`.
- Tests put `experiments/` on `sys.path` via the existing `tests/conftest.py`; run with `uv run --extra dev pytest`.

---

### Task 1: `[bench]` extra + HotpotQA loader

**Files:**
- Modify: `pyproject.toml` (add `[bench]` extra)
- Create: `experiments/benchmarks.py`
- Test: `tests/test_benchmarks.py`

**Interfaces:**
- Produces:
  - `hotpot_to_example(row: dict, split: int) -> crag.CragExample` — pure mapping from one HF row to the contract.
  - `load_hotpotqa(split: int, limit: int | None = None) -> Iterator[crag.CragExample]` — lazy HF streaming loader.
  - `BENCHMARKS: dict[str, callable]` — `{"hotpotqa": load_hotpotqa}` (CRAG stays special-cased in CLIs because it needs a `--data` path).

- [ ] **Step 1: Add the `[bench]` extra to `pyproject.toml`**

Add to `[project.optional-dependencies]` (after the `trigger` entry):

```toml
# Second-benchmark generalization (experiments/benchmarks.py): HuggingFace datasets.
bench = ["datasets>=2.20"]
```

- [ ] **Step 2: Write the failing test (pure mapping, no download)**

`tests/test_benchmarks.py`:

```python
from benchmarks import hotpot_to_example


def _fake_row():
    return {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "hard",
        "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
        "context": {
            "title": ["Ed Wood", "Scott Derrickson", "Distractor A", "Distractor B"],
            "sentences": [
                ["Ed Wood is a 1994 film.", " It starred Johnny Depp."],
                ["Scott Derrickson is an American director."],
                ["Unrelated text one."],
                ["Unrelated text two."],
            ],
        },
    }


def test_hotpot_to_example_maps_contract():
    ex = hotpot_to_example(_fake_row(), split=0)
    assert ex.interaction_id == "5a8b57f25542995d1e6f1371"
    assert ex.query.startswith("Were Scott")
    assert ex.answer == "yes"
    assert ex.question_type == "comparison"
    assert ex.split == 0
    # one passage per context paragraph, in order
    assert len(ex.passages) == 4
    assert ex.passages[0] == "Ed Wood is a 1994 film.  It starred Johnny Depp."
    # gold = indices of the two supporting-fact paragraphs (by title)
    assert ex.gold == {0, 1}
    # gold is the SHIPPED label, not string-matched: "yes" is not in the gold passages
    assert all("yes" not in ex.passages[i].lower() for i in ex.gold)


def test_hotpot_to_example_groundable_property():
    ex = hotpot_to_example(_fake_row(), split=0)
    assert ex.groundable is True  # non-empty gold
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_benchmarks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks'`.

- [ ] **Step 4: Create `experiments/benchmarks.py`**

```python
"""Per-benchmark loaders that emit the crag.CragExample contract, so the whole
stabilization/trigger pipeline runs unchanged. See
docs/superpowers/specs/2026-06-20-second-benchmark-generalization-design.md.
"""
from __future__ import annotations

from typing import Iterator, Optional

from crag import CragExample

_HOTPOT_SPLIT = {0: "validation", 1: "train"}


def hotpot_to_example(row: dict, split: int) -> CragExample:
    """Map one hotpotqa/hotpot_qa (distractor) row to the contract.
    1 context paragraph = 1 passage (paragraphs are already passage-sized), so
    passage index == paragraph index and gold = supporting-fact paragraph indices."""
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    passages = ["".join(sents) if False else " ".join(sents) for sents in sentences]
    gold_titles = set(row["supporting_facts"]["title"])
    gold = {i for i, t in enumerate(titles) if t in gold_titles}
    return CragExample(
        interaction_id=row["id"],
        query=row["question"],
        answer=row["answer"],
        alt_ans=[],
        domain="",
        question_type=row.get("type", ""),       # comparison | bridge
        static_or_dynamic=row.get("level", ""),   # easy | medium | hard
        split=split,
        passages=passages,
        gold=gold,
    )


def load_hotpotqa(split: int, limit: Optional[int] = None) -> Iterator[CragExample]:
    from datasets import load_dataset  # lazy: [bench] extra

    hf_split = _HOTPOT_SPLIT.get(split, "validation")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=hf_split, streaming=True)
    n = 0
    for row in ds:
        ex = hotpot_to_example(row, split)
        if not ex.passages or not ex.query.split() or not ex.gold:
            continue
        yield ex
        n += 1
        if limit and n >= limit:
            break


BENCHMARKS = {"hotpotqa": load_hotpotqa}
```

(Note: the `"".join(...) if False else " ".join(...)` is a deliberate no-op guard kept so a reviewer sees the join is space-joined; the implementer may simplify to `" ".join(sents)`.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_benchmarks.py -v`
Expected: 2 passed.

- [ ] **Step 6: Smoke-test the real loader (one streamed example)**

Run:
```bash
uv run --extra bench python3 -c "import sys; sys.path.insert(0,'experiments'); from benchmarks import load_hotpotqa; ex=next(load_hotpotqa(0, limit=1)); print(len(ex.passages),'passages,',len(ex.gold),'gold, q:',ex.query[:50])"
```
Expected: prints `10 passages, 2 gold, q: ...` (validates the HF schema against the mapping).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml experiments/benchmarks.py tests/test_benchmarks.py
git commit -m "feat: HotpotQA loader emitting the CragExample contract + [bench] extra"
```

---

### Task 2: Single-fire baseline in `train_trigger.py`

**Files:**
- Modify: `experiments/train_trigger.py`
- Test: `tests/test_train_trigger.py`

**Interfaces:**
- Consumes: `analytic_saving`, `hidden_latency_ms` (existing).
- Produces:
  - `fixed_single_fire_eval(t_suf: int, n: int, k: int) -> tuple[int, int]` — fire **once** at word `min(k, n)`; returns `(fire_t, n_calls)` where `n_calls = 1 + (1 if premature else 0)`.
  - `single_fire_frontier(questions: dict, target: str, ks, L, delta, c_waste) -> list[dict]` — per-`k` mean saving / misfire / calls, mirroring `baseline_frontier`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train_trigger.py`:

```python
from train_trigger import fixed_single_fire_eval


def test_fixed_single_fire_eval():
    # fire once at word k=6; t_suf=5 -> 6>=5 correct, 1 call
    assert fixed_single_fire_eval(t_suf=5, n=10, k=6) == (6, 1)
    # k=3 < t_suf=5 -> premature, fire clamped to 3, 2 calls (1 + reflector)
    assert fixed_single_fire_eval(t_suf=5, n=10, k=3) == (3, 2)
    # k beyond query length clamps to n
    assert fixed_single_fire_eval(t_suf=5, n=4, k=8) == (4, 2)  # 4<5 premature
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_train_trigger.py::test_fixed_single_fire_eval -v`
Expected: FAIL with `cannot import name 'fixed_single_fire_eval'`.

- [ ] **Step 3: Add the functions to `experiments/train_trigger.py`**

```python
def fixed_single_fire_eval(t_suf: int, n: int, k: int) -> tuple[int, int]:
    """Fire exactly once at word min(k, n). calls = 1 + 1{premature}."""
    fire = min(k, n)
    premature = fire < t_suf
    return fire, 1 + (1 if premature else 0)


def single_fire_frontier(questions: dict, target: str, ks, L, delta, c_waste) -> list[dict]:
    out = []
    for k in ks:
        savings, premature, calls = [], [], []
        for qr in questions.values():
            target_t = qr[0]["t_suf"] if target == "suf" else qr[0]["t_sc"]
            if target_t is None:
                continue
            nw = qr[0]["n_words"]
            ft, c = fixed_single_fire_eval(target_t, nw, k)
            savings.append(analytic_saving(ft, target_t, nw, L, delta, c_waste))
            premature.append(1 if ft < target_t else 0)
            calls.append(c)
        m = len(savings)
        out.append({"k": k,
                    "mean_saving_ms": sum(savings) / m if m else 0.0,
                    "misfire_rate": sum(premature) / m if m else 0.0,
                    "mean_calls": sum(calls) / m if m else 0.0})
    return out
```

- [ ] **Step 4: Wire `single_fire_frontier` into `main()`'s summary**

In `train_trigger.main()`, after the `baseline = baseline_frontier(...)` line, add:

```python
    single_fire = single_fire_frontier(test_q, "suf", [1, 2, 3, 4, 5, 6, 8], args.L, args.delta, args.c_waste)
```

and add `"single_fire_frontier": single_fire,` to the `summary` dict (next to `"baseline_frontier": baseline,`).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_train_trigger.py -v`
Expected: all pass (existing + the new `test_fixed_single_fire_eval`).

- [ ] **Step 6: Commit**

```bash
git add experiments/train_trigger.py tests/test_train_trigger.py
git commit -m "feat: single-fire fixed baseline for equal-compute trigger comparison"
```

---

### Task 3: `--benchmark` dispatch in `trigger_features.py`

**Files:**
- Modify: `experiments/trigger_features.py`
- Test: `tests/test_trigger_features.py`

**Interfaces:**
- Consumes: `benchmarks.BENCHMARKS`, `crag.load_crag`.
- Produces: `extract(..., benchmark: str = "crag")` dispatches the example source; `main()` gains `--benchmark {crag,hotpotqa}` (`--data` required only for crag).

- [ ] **Step 1: Write the failing test (injected loader, no download)**

Add to `tests/test_trigger_features.py`:

```python
from trigger_features import extract_from_examples, FEATURE_FIELDS
from crag import CragExample


def test_extract_from_examples_counts_and_schema():
    exs = [
        CragExample("q1", "who founded microsoft", "x", [], "", "simple", "", 0,
                    passages=["microsoft was founded by bill gates", "unrelated text"],
                    gold={0}),
    ]
    rows = extract_from_examples(exs, top_k=3, ner_fn=lambda q: [])
    by_q = {}
    for r in rows:
        by_q.setdefault(r["interaction_id"], []).append(r)
    qr = by_q["q1"]
    assert [r["t"] for r in qr] == list(range(1, qr[0]["n_words"] + 1))
    assert set(FEATURE_FIELDS).issubset(qr[0].keys())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_trigger_features.py::test_extract_from_examples_counts_and_schema -v`
Expected: FAIL with `cannot import name 'extract_from_examples'`.

- [ ] **Step 3: Refactor `extract` to split the example source from the per-example work**

In `experiments/trigger_features.py`, replace the body of `extract` so the per-example loop is a reusable function, and add benchmark dispatch:

```python
def extract_from_examples(examples, top_k: int, ner_fn=spacy_ner) -> list[dict]:
    out = []
    for ex in examples:
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


def _examples(benchmark: str, data: Optional[str], split: int, limit: Optional[int]):
    if benchmark == "crag":
        return load_crag(data, split=split, limit=limit)
    from benchmarks import BENCHMARKS
    return BENCHMARKS[benchmark](split, limit=limit)


def extract(data, split, top_k, ner_fn=spacy_ner, limit=None, benchmark="crag") -> list[dict]:
    return extract_from_examples(_examples(benchmark, data, split, limit), top_k, ner_fn)
```

Then add to `main()`'s argparse (before `args = ap.parse_args()`):

```python
    ap.add_argument("--benchmark", default="crag", choices=["crag", "hotpotqa"])
```

and pass it through: `rows = extract(args.data, args.split, args.top_k, limit=args.limit, benchmark=args.benchmark)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_trigger_features.py -v`
Expected: all pass (existing fixture test still green — `extract` still works for crag).

- [ ] **Step 5: Commit**

```bash
git add experiments/trigger_features.py tests/test_trigger_features.py
git commit -m "feat: --benchmark dispatch in trigger_features (crag default, hotpotqa)"
```

---

### Task 4: `benchmark_stats.py` — clean vs string-grounded φ_suf (the bias arm)

**Files:**
- Create: `experiments/benchmark_stats.py`
- Test: `tests/test_benchmark_stats.py`

**Interfaces:**
- Consumes: `stabilization.stabilization`, `crag.gold_passage_ids`, `crag.CragExample`, `benchmarks.BENCHMARKS`.
- Produces:
  - `dual_stab(ex, top_k) -> dict | None` — per-question row with `t_suf_clean`/`phi_suf_clean` (from `ex.gold`) and `t_suf_string`/`phi_suf_string` (from `gold_passage_ids`), plus `n_words`, `question_type`.
  - `summarize_dual(rows) -> dict` — median/mean φ_suf and `t_suf=1` share under each grounding, on retrieved-gold rows.
  - `main()` CLI: `--benchmark --split --top-k --out --summary-out`.

- [ ] **Step 1: Write the failing test**

`tests/test_benchmark_stats.py`:

```python
from benchmark_stats import dual_stab, summarize_dual
from crag import CragExample


def test_dual_stab_clean_vs_string():
    # clean gold = passage 1 (shipped). The short answer "1" string-grounds to BOTH
    # passages, so string-grounding can surface gold at an earlier prefix than clean.
    ex = CragExample("q1", "how many moons does earth have", "1", [], "", "simple", "", 0,
                     passages=["the planet earth orbits the sun",
                               "earth has 1 moon called luna"],
                     gold={1})
    r = dual_stab(ex, top_k=1)
    assert r is not None
    assert r["t_suf_clean"] is not None
    # string grounding marks any passage containing "1" as gold (here passage 1 too,
    # plus possibly others); both are defined here
    assert r["t_suf_string"] is not None
    assert r["n_words"] == 6


def test_summarize_dual_reports_both():
    rows = [
        {"retrieved_gold_clean": True, "phi_suf_clean": 0.4, "t_suf_clean": 2,
         "retrieved_gold_string": True, "phi_suf_string": 0.1, "t_suf_string": 1,
         "n_words": 5, "question_type": "comparison"},
        {"retrieved_gold_clean": True, "phi_suf_clean": 0.5, "t_suf_clean": 3,
         "retrieved_gold_string": True, "phi_suf_string": 0.17, "t_suf_string": 1,
         "n_words": 6, "question_type": "bridge"},
    ]
    s = summarize_dual(rows)
    assert s["clean"]["phi_suf_median"] == 0.45
    assert s["string"]["phi_suf_median"] == 0.135
    assert s["string"]["t_suf_eq_1_rate"] == 1.0
    assert s["clean"]["t_suf_eq_1_rate"] == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/test_benchmark_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark_stats'`.

- [ ] **Step 3: Create `experiments/benchmark_stats.py`**

```python
"""Clean vs string-grounded phi_suf on a benchmark with SHIPPED gold passages.
The clean arm uses ex.gold; the string arm re-derives gold via the CRAG matcher,
so the gap measures the over-grounding bias on independent labelled data.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from typing import Optional

from crag import CragExample, gold_passage_ids
from stabilization import stabilization


def dual_stab(ex: CragExample, top_k: int) -> Optional[dict]:
    clean = stabilization(ex.query, ex.passages, ex.gold, top_k=top_k)
    if clean is None:
        return None
    gold_string = gold_passage_ids(ex.answer, ex.alt_ans, ex.passages)
    strung = stabilization(ex.query, ex.passages, gold_string, top_k=top_k)
    return {
        "interaction_id": ex.interaction_id,
        "question_type": ex.question_type,
        "n_words": clean.n_words,
        "retrieved_gold_clean": clean.retrieved_gold,
        "t_suf_clean": clean.t_suf,
        "phi_suf_clean": clean.phi_suf,
        "retrieved_gold_string": strung.retrieved_gold if strung else False,
        "t_suf_string": strung.t_suf if strung else None,
        "phi_suf_string": strung.phi_suf if strung else None,
    }


def _cell(rows, gold_key, phi_key, t_key) -> dict:
    sub = [r for r in rows if r[gold_key] and r[phi_key] is not None]
    if not sub:
        return {"n": 0}
    phi = [r[phi_key] for r in sub]
    return {"n": len(sub),
            "phi_suf_mean": round(st.mean(phi), 4),
            "phi_suf_median": round(st.median(phi), 4),
            "t_suf_eq_1_rate": round(sum(1 for r in sub if r[t_key] == 1) / len(sub), 4)}


def summarize_dual(rows: list[dict]) -> dict:
    return {
        "clean": _cell(rows, "retrieved_gold_clean", "phi_suf_clean", "t_suf_clean"),
        "string": _cell(rows, "retrieved_gold_string", "phi_suf_string", "t_suf_string"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="hotpotqa", choices=["hotpotqa"])
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/hotpotqa/dual_stab.csv")
    ap.add_argument("--summary-out", default="results/hotpotqa/dual_stab.summary.json")
    args = ap.parse_args()

    from benchmarks import BENCHMARKS
    rows = [r for ex in BENCHMARKS[args.benchmark](args.split, limit=args.limit)
            if (r := dual_stab(ex, args.top_k))]
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(args.summary_out, "w") as f:
        json.dump({"params": {"benchmark": args.benchmark, "split": args.split, "top_k": args.top_k},
                   "n_questions": len(rows), "dual": summarize_dual(rows)}, f, indent=2)
    s = summarize_dual(rows)
    print(f"clean: {s['clean']}  | string: {s['string']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_benchmark_stats.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `uv run --extra dev pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/benchmark_stats.py tests/test_benchmark_stats.py
git commit -m "feat: benchmark_stats clean-vs-string-grounded phi_suf (over-grounding bias arm)"
```

---

### Task 5: Real HotpotQA run (controller-executed data pass)

**Files:**
- Create (outputs): `results/hotpotqa/*`

This task downloads HotpotQA and runs the three measurements. It is **run by the controller**, not a code subagent (it is data-heavy and long; no new code). Steps:

- [ ] **Step 1: Generate per-word feature tables (clean gold) for both splits**

```bash
uv run --extra bench --extra trigger experiments/trigger_features.py --benchmark hotpotqa --split 1 --top-k 3 --out results/hotpotqa/trigger_features.split1.csv
uv run --extra bench --extra trigger experiments/trigger_features.py --benchmark hotpotqa --split 0 --top-k 3 --out results/hotpotqa/trigger_features.split0.csv
```

- [ ] **Step 2: Run the bias arm (clean vs string-grounded φ_suf)**

```bash
uv run --extra bench experiments/benchmark_stats.py --benchmark hotpotqa --split 0 --top-k 3 \
  --out results/hotpotqa/dual_stab.csv --summary-out results/hotpotqa/dual_stab.summary.json
```

Expected: prints `clean: {...}  | string: {...}` — compare median φ_suf and t_suf=1 share between the two (the bias measurement).

- [ ] **Step 3: Train + evaluate the trigger vs single-fire + multi-fire baselines**

```bash
uv run --extra trigger experiments/train_trigger.py \
  --train results/hotpotqa/trigger_features.split1.csv \
  --test results/hotpotqa/trigger_features.split0.csv \
  --summary-json results/hotpotqa/trigger.summary.json --plot-out results/hotpotqa/trigger_frontier.png
```

- [ ] **Step 4: Sanity-check + commit results**

Inspect `results/hotpotqa/dual_stab.summary.json` (does string-grounding shift φ_suf earlier / inflate t_suf=1 vs clean?), `trigger.summary.json` (Spearman, ablation, trigger vs single-fire frontier). Then:

```bash
git add results/hotpotqa/
git commit -m "results: HotpotQA generalization (clean phi_suf, over-grounding bias arm, trigger vs single-fire baseline)"
```

---

## Self-Review

**Spec coverage:**
- §2 one-contract reuse → Task 1 (loader emits CragExample); Tasks 3–4 consume it unchanged. ✓
- §3 HotpotQA distractor, paragraph-as-passage, shipped gold → Task 1 (`hotpot_to_example`, mapping + test asserting gold≠string-match). ✓
- §4(a) clean φ_suf → Task 4 `dual_stab` clean arm + Task 5 features. ✓
- §4(b) string-grounding bias arm → Task 4 (`dual_stab` string arm, `summarize_dual`) + Task 5 Step 2. ✓
- §4(c) trigger vs single-fire baseline → Task 2 (single-fire) + Task 3 (features --benchmark) + Task 5 Step 3. ✓
- §5 `[bench]` extra, `benchmarks.py`, `benchmark_stats.py`, single-fire in train_trigger, `results/hotpotqa/` → Tasks 1,2,4. ✓
- §6 HotpotQA first → this plan is HotpotQA only; **NQ is a deliberate follow-up plan** (noted gap, not an omission).
- §8 per-question passages (no open retrieval) → Task 1 paragraph-as-passage. ✓

**Placeholder scan:** none — every code/test step is complete. (The `if False else` join guard in Task 1 Step 4 is intentional and annotated.)

**Type consistency:** `hotpot_to_example`/`load_hotpotqa`/`BENCHMARKS` names match across Tasks 1/3/4. `fixed_single_fire_eval` returns `(fire_t, calls)` consistent between Task 2 def and test. `extract_from_examples(examples, top_k, ner_fn)` signature matches Task 3 def and test. `dual_stab` row keys match between Task 4's producer, its `summarize_dual` consumer, and the tests.

**Known follow-ups (not gaps):** NQ loader + run is a separate plan (per spec §6 sequencing); the MuSiQue/2Wiki fallback is selected there if NQ acquisition stalls.
