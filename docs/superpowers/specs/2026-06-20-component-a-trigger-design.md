# Design — Paper 2 Component A: The Classifier Trigger

**Status:** Approved (design phase). **Date:** 2026-06-20.
**Parent spec:** [docs/paper_2_proposal.md](../../paper_2_proposal.md) (Component A).
**Scope:** This is the *first* sub-project of Paper 2. Components B–E are out of scope
and get their own spec→plan cycles. Build-ready; offline-analytic eval only (no LLM,
no async harness — those are Components B/D).

---

## 1. Goal & contract

A model that, at each word position *t* of a streaming query, predicts **P(t ≥ t_suf)**
from live-observable features, **fires once** when P first crosses a threshold τ, and
beats the fixed-interval baseline on a **latency–compute frontier** without raising the
mis-fire rate.

- **Train:** CRAG split 1 (1,335 questions) — retrieved-gold subset supplies all supervision.
- **Test:** CRAG split 0 (1,371 questions). Reported **once**; never touched for model selection.
- **No leakage:** τ and hyperparameters chosen by cross-validation *within* split 1.
- **CPU-only, training-free at inference** (model is logistic regression / a small GBDT).

The classifier never optimizes perceived-latency saving directly — saving depends on *n*
(unknown live) and is a downstream **evaluation** quantity. The model optimizes the
live-observable proxy P(t ≥ t_suf).

---

## 2. Grounding in the existing code

| Existing thing | Location | How Component A uses it |
| :-- | :-- | :-- |
| Prefix retrieval sweep | `experiments/stabilization.py` `_prefix_sequence()` | Refactor into a public `prefix_records()` and reuse verbatim — no duplicated retrieval |
| `t_suf`, `t_sc`, `retrieved_gold`, `volatility` | `stabilization.py` `stabilization()` → `Stab` | Source of the label (`t_suf`) and of the t_sc secondary target |
| `H` bound | `stabilization.py` `hidden_latency_ms(t_star, n_words, L_ms, delta_wps)` | The analytic saving function (§6) |
| CRAG loading, `question_type`, gold grounding | `experiments/crag.py` `load_crag()`, `gold_passage_ids()` | Iterate questions for both splits |
| Fixed-interval trigger | `experiments/streaming_rag.py` (`trigger_interval_words`) | Defines the baseline firing rule (re-implemented offline in §6) |
| Per-question CSV schema | `run_study.py` `FIELDS` | The *new* per-word table is a separate artifact (the per-question CSV has no per-word rows) |

**Key fact:** `stabilization()` already computes the per-prefix top-1 sequence but discards
it after deriving scalars. Component A's retrieval-side features are exactly that sequence,
surfaced per-word.

---

## 3. New modules & artifacts (small, focused units)

- **`experiments/trigger_features.py`** — per-word feature/label extraction.
  CLI: `--data <bz2> --split {0,1} --top-k 3 --out results/trigger_features.split{N}.csv`.
  One row per `(interaction_id, t)`.
- **`experiments/train_trigger.py`** — fit LogReg + GBDT, sweep τ → frontier, ablation,
  metrics. CLI: `--train results/trigger_features.split1.csv --test results/trigger_features.split0.csv
  --summary-json results/trigger.summary.json --plot-out results/trigger_frontier.png`.
- **Refactor `experiments/stabilization.py`** — extract `prefix_records(query, passages, top_k, make_retriever) -> (seq, n)`
  (public), used by both `stabilization()` and `trigger_features.py`.
- **`pyproject.toml`** — new optional extra `[trigger]` = `scikit-learn`, `spacy` (+ a note to
  `python -m spacy download en_core_web_sm`). Core stays zero-dependency.
- **`experiments/audit_labels.py`** — sample ~100 retrieved-gold questions → adjudication CSV (§7).

### Output artifacts (the reproducible record)
- `results/trigger_features.split0.csv`, `results/trigger_features.split1.csv`
- `results/trigger.summary.json` — frontier, chosen operating point, ablation, importances,
  per-population metrics, `params` provenance block.
- `results/trigger_frontier.png` — latency-saving vs retrieval-calls Pareto, trigger vs fixed-interval.
- `results/label_audit.csv` (+ a precision estimate in the summary).

---

## 4. Features (per *t*, all live-observable)

| Feature | Source | Definition |
| :-- | :-- | :-- |
| `top1_stable_streak` | prefix top-1 seq | consecutive prefixes ending at *t* with identical top-1 id (≥1); the likely dominant signal |
| `top1_changed` | prefix top-1 seq | 1 if top-1 at *t* ≠ top-1 at *t*−1 (0 at t=1) |
| `t` | token stream | absolute word count — **never** t/n (*n* unknown live) |
| `named_entity_detected` | spaCy | 1 once *t* ≥ first-NE word offset, else 0 |
| `words_since_first_ne` | spaCy | `max(0, t − first_ne_pos)`; encodes the entity-anchor hypothesis (0 if no NE yet) |
| `question_word_type` | leading token(s) | who/what/when/where/which/other, one-hot |

**NER fidelity caveat (documented limitation).** spaCy (`en_core_web_sm`) runs **once on the
full query**; detection at *t* uses each entity's full-sentence word offset
(`detected ⇔ t ≥ offset`). This is faithful for *timing* but uses full-sentence boundaries —
a small optimism a deployed incremental-NER system would not have. The headline rests on
`top1_stable_streak` (exactly live); the entity features are what the ablation interrogates.
An incremental-NER robustness check is named as future work.

---

## 5. Labels & populations

- **Label** `1[t ≥ t_suf]`, defined **only on retrieved-gold questions** (the ~21% where
  `t_suf` is not None). These questions supply all supervision. Each contributes `n_words`
  per-word rows.
- **Non-retrieved-gold rows** (label all-0, no positive) are **held out of training** but
  retained in the feature CSV (flagged `retrieved_gold=False`) to:
  1. report **deployment behavior** on the ungroundable majority on the test split, and
  2. fit a **secondary `t_sc`-target model** (`1[t ≥ t_sc]`, always defined) — the safety
     check that the trigger does not fire pathologically (never / constantly) on that majority.
- The headline trigger is the **t_suf** model; the t_sc model is reported as the safety/secondary
  result. If the two disagree sharply on the ungroundable set, that bounds how far a
  sufficiency-trained trigger generalizes (a reportable limit).

---

## 6. Offline-analytic evaluation

Decode each test question to a single **fire_t** = smallest *t* with P(t) ≥ τ (or "never fires"
if P stays below τ). Realized-saving model, built on `hidden_latency_ms()` at canonical
(L=600 ms, δ=3 w/s):

- **Correct fire** (`fire_t ≥ t_suf`): `saving = H(fire_t) = min(L, (n−fire_t)/δ·1000)`.
  Maximized at `fire_t = t_suf`; later fires save less.
- **Premature fire** (`fire_t < t_suf`): speculative call misses gold; Reflector re-fires at
  recovery (t_suf) → `saving = H(t_suf) − C_waste`, with **`C_waste = L` by default** (one
  wasted retrieval round). This is what reproduces net-negative outcomes. **Sensitivity-checked**
  over `C_waste ∈ {0.5L, L}`; calibrated against Component D's real harness numbers later.
- **Never fires:** `saving = 0`, counts as a fully-missed opportunity.

**Frontier.** Sweep τ ∈ [0,1]; per τ compute on the test set:
- mean saving (ms), mis-fire rate (% fires premature), net-negative rate (% saving ≤ 0),
- **retrieval-calls/question** = `1 + 1{premature}` for the trigger,
- median saving at correct fires, **Spearman ρ(fire_t, t_suf)** across questions.

**Baseline.** Fixed-interval trigger fires at `interval, 2·interval, …`, issuing a speculative
retrieval at each until one lands at `t ≥ t_suf` (the Reflector then accepts and stops). So its
**retrieval-calls/question ≈ ⌈t_suf / interval⌉** (capped at `⌈n / interval⌉` when it never
succeeds) — the "many speculative retrievals" cost — and its realized saving is `H(first fire ≥ t_suf)`.
We **sweep the baseline's `interval`** (default 2, the harness `trigger_interval_words`) so the
comparison is frontier-vs-frontier, not point-vs-frontier. Headline comparison rule:
**match the baseline's mis-fire rate, beat it on mean saving — at equal-or-lower retrieval-calls.**

**Populations.** All metrics reported separately for retrieved-gold (target = t_suf) and
t_sc-fallback (target = t_sc).

---

## 7. Label-quality audit

`audit_labels.py` samples ~100 retrieved-gold questions (seeded) and emits `results/label_audit.csv`
with the question, the grounded gold passage text, and a blank `is_answer_bearing` column for
adjudication (manual, or LLM-judge with human spot-checks). The resulting **label-precision
estimate** is recorded in `trigger.summary.json` and bounds the classifier's achievable ceiling
(so reviewers can separate trigger error from label error).

---

## 8. Train/test discipline

- Generate `trigger_features.split1.csv` (train) and `trigger_features.split0.csv` (test) — split 1
  labels do **not** exist yet and require running `trigger_features.py --split 1` (CPU, hours).
- Model selection (τ, GBDT depth/estimators, LogReg class-weight) via k-fold CV **within split 1**.
- Split 0 is scored once, at the end. Clean train/test separation is a stated strength.

---

## 9. Deferred (explicitly out of scope for Component A)

- Real async-harness *measured* perceived-latency (ms) → **Component D**.
- End-to-end generation & answer quality → **Component B**.
- Reranker / cross-retriever TIS × quality → **Component C**.

---

## 10. Component-level kill-criteria

- **If neither model beats fixed-interval on the frontier:** the ablation + Spearman still
  characterize *why* (label noise via the §7 audit, or genuinely weak features), and metric
  validity (Component C) becomes the fallback headline — consistent with the proposal's venue fallback.
- **If the NER timing approximation drives the entity result:** report it honestly, lean the
  headline on `top1_stable_streak` (exactly live), and name incremental NER as future work.
- **If label precision (§7) is low:** the achievable accuracy ceiling is reported as the binding
  constraint, not hidden.

---

## 11. Open items intentionally fixed here (no further decisions needed)

- Entity features: spaCy NER, **core feature** from the start (not ablation-only).
- Model: **LogReg primary** (interpretable headline) **+ GBDT secondary** (nonlinear headroom).
- Eval fidelity: **offline analytic now**, real harness deferred to D.
- Premature-fire penalty: `C_waste = L` (sensitivity over {0.5L, L}).
