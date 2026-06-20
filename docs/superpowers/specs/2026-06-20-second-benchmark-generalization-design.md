# Design — Second-Benchmark Generalization (HotpotQA + NQ)

**Status:** Approved (design phase). **Date:** 2026-06-20.
**Parent:** Paper 2, the "second benchmark" component (proposal Component E), reprioritized
ahead of the live-trigger work after the CRAG grounding finding.
**Scope:** One sub-project. HotpotQA built and validated first; NQ second (with a fallback).
Everything reuses the existing CRAG pipeline through one contract.

---

## 1. Why this, why now

On CRAG we established two things that make a single-benchmark result untrustworthy:
- **Over-grounding artifact** ([[grounding-overcounts-phi-suf]]): CRAG ships no gold-passage
  label, so `gold_passage_ids` string-matches the answer; short/ubiquitous answers match many
  passages, `t_suf` collapses to 1 for ~49% of retrieved-gold questions, and `phi_suf` is biased
  low (median 0.14 vs ~0.35 on precisely-grounded questions).
- **Data dominates system** ([[component-a-negative-result]]): the trigger's per-query signal is
  invisible on the artifact-laden full set (Spearman −0.09) and only appears on the clean subset
  (+0.34); even then CRAG's short, tightly-clustered `t_suf` (median ~4) leaves a fixed single-fire
  baseline very hard to beat.

A second (and third) benchmark with **clean, shipped gold-passage labels** and a **different
query-length regime** separates "is there a real effect" from "is CRAG degenerate." HotpotQA
(long multi-hop) and NQ (short real queries) bracket the regime.

**Success criteria (what each result will tell us):**
- `phi_suf` (clean gold) early on Hotpot/NQ → early-stabilization **generalizes**, not a CRAG artifact.
- `phi_suf`(string-grounded) ≪ `phi_suf`(clean) on the same questions → the over-grounding bias is
  **real and replicated** on independent data (validates the Paper 1 correction).
- Trigger signal (Spearman, entity-position ablation, vs single-fire baseline) **stronger where
  `t_suf` spreads** (Hotpot) than where it clusters (NQ) → **query length predicts where the system
  helps** — the data-vs-system separation, made quantitative.

---

## 2. Architecture — one contract, total reuse

The entire pipeline (`prefix_records`, `stabilization`, `trigger_features`, `train_trigger`,
`grounding_precision`) consumes a `CragExample`: `.query`, `.passages: list[str]`, `.gold: set[int]`,
plus `.answer`, `.alt_ans`, `.question_type`, `.interaction_id`, `.split`. So the only new code is
**thin per-benchmark loaders that emit that exact shape**; nothing downstream changes.

New module **`experiments/benchmarks.py`**:
- `load_hotpotqa(split, ...) -> Iterator[CragExample]`
- `load_nq(split, ...) -> Iterator[CragExample]`  (built second)
- A small registry so CLIs can take `--benchmark {crag,hotpotqa,nq}` and dispatch to the right loader
  (CRAG stays the default via `crag.load_crag`).

Each loader produces **two gold sets per question** (see §4): the benchmark's clean gold and a
string-grounded gold. The clean set populates `.gold`; the string-grounded set is computed on demand
by the bias arm via the existing `crag.gold_passage_ids` over `.answer`/`.alt_ans`/`.passages`.

**New dependency:** a `[bench]` optional extra = `datasets` (HuggingFace) for data loading. Core stays
zero-dependency; retrieval remains BM25/CPU, unchanged.

---

## 3. Benchmarks & data acquisition

**HotpotQA (primary, built first).** HF `hotpot_qa`, config `distractor`. Each example ships
`context` = 10 paragraphs (title + sentences) and `supporting_facts` = the gold titles. Map:
`passages` = the 10 paragraphs (chunked with the existing `chunk_text` for parity), `gold` = indices
of the supporting-fact paragraphs (**clean, no string-matching**). `question_type` = `"multihop"`
(uniform) or the dataset's `type`/`level` if present. Long multi-hop queries → `t_suf` should spread.
Small download; ships passages+gold directly → de-risks the whole reuse architecture.

**NQ (second).** HF `natural_questions`, validation split. `passages` = the document's paragraphs
(chunked), `gold` = the long-answer paragraph index (**clean**). Short real queries → the contrast
regime. **Risk:** the full document data is large (tens of GB) and parsing the HTML/token spans is
fiddly. Mitigation: build HotpotQA fully first; attempt NQ second on the validation split only.

**NQ fallback (if acquisition/parse proves impractical):** substitute **MuSiQue or 2WikiMultihopQA**
— both ship per-question contexts with gold supporting paragraphs (easy, like Hotpot). Explicit
tradeoff to state in the paper: these are multi-hop, so we **lose the short-query contrast** NQ
provided; the generalization claim holds but the "query-length predicts where the system helps" axis
weakens. Do not swap silently.

---

## 4. The three measurements (each reuses existing code)

**(a) Clean `phi_suf`.** Run `stabilization(query, passages, gold_clean, top_k=3)` per question →
`t_suf`, `phi_suf`, the early-stabilization distribution from *real* gold passages. The honest,
artifact-free measurement CRAG could not provide. Report `phi_suf`/`phi_sc` distributions per
benchmark, same statistics as Paper 1.

**(b) Grounding-bias arm.** On the *same* questions, also compute `gold_string =
gold_passage_ids(answer, alt_ans, passages)` and run `stabilization` with it → `t_suf_string`,
`phi_suf_string`. Report the paired gap `phi_suf_clean` vs `phi_suf_string` and the `t_suf=1` share
under each. A large early-shift under string grounding **replicates and quantifies the CRAG
over-grounding bias on independent, labelled data** — the validation of the Paper 1 correction.
Reuse `experiments/grounding_precision.py`'s density-sweep machinery for the precision view.

**(c) Trigger vs single-fire baseline.** Run `trigger_features` + `train_trigger` per benchmark,
targeting **clean gold**, with a train/test split (Hotpot has train/validation; NQ uses
validation split partitioned, or train if available). Report Spearman(fire, `t_suf`), the
entity-position ablation, and the frontier vs **both** baselines. This requires folding the
**single-fire fixed baseline** (prototyped in the CRAG analysis: fire once at word `k`, sweep `k`)
into `train_trigger` so the **equal-compute** comparison is in the harness, not ad hoc.

---

## 5. New code & artifacts

- `experiments/benchmarks.py` — loaders + registry (HotpotQA first, NQ second).
- `experiments/train_trigger.py` — add `fixed_single_fire_eval(t_suf, n, k)` + a single-fire
  baseline frontier alongside the existing multi-fire `baseline_frontier`; surface both in the summary.
- A `--benchmark` flag on the feature/stat CLIs (`trigger_features.py`, `run_study.py` or a thin
  wrapper) dispatching to the registry.
- `pyproject.toml` — `[bench]` extra (`datasets`).
- Outputs under `results/<benchmark>/`: `*.summary.json` (clean), `*.string.summary.json` (bias arm),
  `trigger.<benchmark>.summary.json`, feature CSVs. Mirror the CRAG artifact conventions.

---

## 6. Sequencing

1. **HotpotQA end-to-end** (loader → clean `phi_suf` → bias arm → trigger + single-fire baseline).
   Proves the reuse architecture and yields the first generalization + bias-validation result.
2. **NQ** (loader → same three measurements). If NQ acquisition stalls, switch to the MuSiQue/2Wiki
   fallback with the contrast-loss caveat.
3. Cross-benchmark synthesis: does early-stabilization generalize, does the bias replicate, does
   `t_suf` spread (and trigger value) track query length.

---

## 7. Risks / kill-criteria

- **NQ data acquisition** (the main risk) → HotpotQA-first sequencing + MuSiQue/2Wiki fallback (§3).
- **Loaders don't faithfully emit the contract** (e.g., HotpotQA gold-title→paragraph index mapping
  is wrong) → unit-test each loader on a tiny fixture asserting gold paragraphs actually contain the
  answer; the bias arm's clean-vs-string agreement is a cross-check.
- **Clean `phi_suf` also turns out early-and-trivial** (e.g., HotpotQA gold paragraphs are findable
  from the first entity too) → that is itself a result (early-stabilization is robust); report it.
- **Trigger still loses to single-fire even where `t_suf` spreads** → the honest conclusion is that
  the per-query signal, though real, does not beat fixed timing on these workloads either, and the
  paper's center of gravity stays on the measurement (Paper 1 correction + cross-benchmark
  generalization) rather than the live trigger.

---

## 8. Out of scope (own later cycles)

- Cross-retriever TIS × quality (the original Component C) and end-to-end answer generation
  (Component B). This sub-project is measurement + the existing trigger on new data only.
- Full-corpus (open) retrieval for NQ; we use the per-question page/distractor passage sets to keep
  the CPU-only, per-question structure that the whole pipeline assumes.
