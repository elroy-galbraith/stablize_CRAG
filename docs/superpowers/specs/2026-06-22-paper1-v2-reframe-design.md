# Paper 1 → v2 reframe: tool-intent stabilization is operationalization-dependent

**Date:** 2026-06-22
**Form:** in-place v2 revision of `paper/main.tex` (one arXiv lineage, new version)
**Status:** design approved (in-place v2 · new spine · compute global headroom)

## Problem

Paper 1 v1 ("When Does Streaming Tool Use Help?") reports that tool-intent
stabilization is *often early* — the retriever locks onto the answer-bearing
passage well before the query finishes, so most tool latency can be hidden behind
the user's remaining input. v1 already carries one hedge (the grounding-precision
correction: median `phi_suf` 0.14 → 0.35 on precisely-grounded items) and even
claims a dense-retriever arm shows the effect "is not a BM25 lexical artifact."

New global-corpus evidence contradicts the headline at the thesis level, not just
the magnitude. When prefixes are retrieved against a **realistic global corpus**
instead of the per-question candidate pool CRAG ships, stabilization is **late**,
moves **monotonically later as the corpus grows** (a dose-response), and is
**retriever-general** (BM25 ≈ dense). The early-stabilization picture is largely an
artifact of measuring against passages that were pre-selected by the *full* query.

## Thesis (v2)

Tool-intent stabilization is **operationalization-dependent**: its value is set as
much by how the retrieval corpus is constructed as by the query itself. Under a
per-question candidate pool, stabilization looks early; under a realistic global
corpus it is late, and the latency headroom that motivates streaming RAG largely
collapses. The TIS framework (`t_sc`, `t_suf`, `phi`, volatility `V`, the bound `H`)
remains sound — what changes is the empirical answer and the caveat that the answer
is not corpus-invariant. The contribution becomes a **measurement caution**: report
TIS against the corpus the deployed system will actually retrieve over.

## Evidence base (all already committed; no new experiments)

Numbers are medians of `phi_suf` and the `t_suf == 1` share, k=3, from
`results/global/` and the v1 macros in `paper/main.tex`.

| Operationalization | median phi_suf | t_suf=1 | source |
|---|---|---|---|
| CRAG per-question (v1, string-grounded) | 0.14 | 49% | `\phiSufMed`, `\tSufOneAll` |
| CRAG per-question (v1, precisely-grounded) | 0.35 | 14% | `\phiSufCleanMed`, `\tSufOneClean` |
| Global NQ, gold+1k | 0.571 | 3.3% | `results/global/confirm/nq_bm25_1000.summary.json` |
| Global NQ, gold+10k | 0.625 | 2.9% | `results/global/confirm/nq_bm25_10000.summary.json` |
| Global NQ, ~1M prefix | 0.75 | 0.7% | `results/global/nq_tsuf.1m.summary.json` |
| Global FiQA, gold+10k, BM25 | 0.636 | 2.8% | `results/global/confirm/fiqa_bm25_10k.summary.json` |
| Global FiQA, gold+10k, **dense** | 0.625 | 1.5% | `results/global/confirm/fiqa_dense_10k.summary.json` |

Three independent confirmations: (1) **dose-response** — monotone with corpus size
on NQ; (2) **two benchmarks** — NQ and FiQA agree; (3) **retriever-general** — on the
*identical* FiQA gold+10k corpus, dense (0.625) ≈ BM25 (0.636); dense surfaces gold
for more queries (n=481 vs 364) but at the same late point, so it generalizes the
artifact rather than rescuing the early claim.

Note on the global-corpus `perq` arm: `global_corpus.py` also computes a per-question
arm (top-100 of the global corpus ∪ gold). It is an *intermediate* control — later
than CRAG's tiny native pool, earlier than full-global — useful to show pool size
monotonically shifts stabilization, but it is **not** the paper's primary contrast.
The headline contrast is **CRAG-native per-question (v1)** vs **global corpus**.

## Document changes (`paper/main.tex`, `paper/results.tex`, macros)

### 1. Title + abstract
- Retitle away from "When Does Streaming Tool Use Help?". Offer 2–3 options in the
  draft; working candidate: *"Tool-Intent Stabilization Is an Artifact of the
  Retrieval Corpus: A Cautionary Measurement Study in Streaming RAG."*
- Rewrite the abstract around operationalization-dependence and the headroom
  collapse. **Delete** the v1 sentence asserting the dense arm proves the effect is
  "not a BM25 lexical artifact" — the global dense result reverses its import.

### 2. Introduction
- Replace the "large early-stabilizing mass with a thin late tail" framing with the
  corpus-dependence result as the lead. Keep the PlayStation example (it still
  motivates *why* stabilization point matters) but recast the empirical claim.

### 3. Problem Formalization
- Make the retrieval corpus an explicit parameter of `t_suf` (and hence `phi_suf`):
  `t_suf` is defined *with respect to a corpus C*. State that v1's results are
  `t_suf(· ; C_perq)` and the new results are `t_suf(· ; C_global)`.
- `t_sc`, `V`, and the bound `H` definitions are unchanged.

### 4. Method
- Add the global-corpus protocol: gold-guaranteed subsampling (corpus = all gold ∪
  N distractors, reservoir-sampled), dose-response over N, BM25 (`bm25s`) and dense
  (all-MiniLM-L6-v2, cosine), on NQ + FiQA. Reference `experiments/global_corpus.py`.

### 5. NEW Results spine — global vs per-question
- New section leading the results: the table above + the three confirmations.
- The `t_suf == 1` collapse (≈50% → ≤3%) gets explicit treatment: "one token is
  almost never sufficient under realistic retrieval."

### 6. RQ1–RQ4 demoted to "the per-question view"
- Keep all four RQ subsections and their numbers, relabeled as the per-question
  (optimistic) operationalization, not the headline. RQ4's type effect stays as a
  within-per-question observation.

### 7. Headroom: recompute streamable fraction under the global corpus
- **New analysis (cheap, arithmetic over committed CSVs):** compute the streamable
  fraction at L=600, δ=3 using `t_suf_global`/`n_words` from
  `results/global/confirm/*.csv` and `results/global/nq_tsuf.1m.csv`. Report
  side-by-side with v1's 73.9% to quantify the collapse. Add a small script
  `experiments/global_headroom.py` (see plan) that emits the global streamable
  fraction + a summary JSON under `results/global/`. Note the θ caveat: v1's RQ2
  sweeps a coverage threshold θ, but the global `t_suf` uses a hard sufficiency
  criterion (gold appears in top-k), so θ is implicit, not a free knob, in the
  global arm — compare at matched (L, δ) and state this explicitly rather than
  implying a θ=0.8 cell for the global numbers.
- **RQ3 (H-bound replay):** keep as-is but scope it explicitly — it validates the
  bound *mechanism* under the per-question (optimistic) operationalization. State
  that a global-corpus replay is future work (the async harness retrieves over
  per-question passages; re-plumbing it to a 10k+ global index is out of scope).

### 8. Conclusion + Limitations
- Lead the conclusion with the measurement caution. Keep the framework as the
  durable contribution. In Limitations: name the per-question pool as the source of
  the v1 over-estimate; note the global corpus is itself a BEIR construction (NQ/FiQA
  qrels), not the deployed system's corpus, so the true number is system-specific.

## Components / interfaces

- **`experiments/global_headroom.py`** (new, stdlib-only): read a global CSV, compute
  per-row hidden-latency `H = min(L, max(0, (n − t_suf)/δ))` and the streamable
  fraction (rows with residual > 0), at configurable (L, δ). Emit CSV + summary JSON.
  Mirrors the RQ2 arithmetic already in `run_study.py` so the two are comparable.
  Testable in isolation: feed a tiny CSV, assert the fraction.
- **`paper/main.tex` macros block**: add `\phiSufGlobal*`, `\tSufOneGlobal`,
  dose-response points, `\streamPctGlobal`, FiQA + dense numbers. One macro per
  number, sourced from the JSON files named above (provenance comment per macro, as
  the existing block already does).
- **Existing code unchanged**: `global_corpus.py`, `stabilization.py`, `crag.py` are
  not modified — the v2 is a *writing + light-analysis* task, not a code change.

## Testing / verification

- `global_headroom.py`: pytest with a hand-built tiny CSV (known `t_suf`/`n`) →
  assert `H` per row and the streamable fraction; an empty-CSV guard.
- Macro provenance: every new macro carries a comment naming its source JSON; a
  reviewer can re-derive each from the committed artifacts.
- LaTeX builds (`latexmk`/`pdflatex`) without undefined-reference errors.
- No claim in the abstract/intro/conclusion lacks a backing committed number.

## Out of scope (deferred, named so they are not silently dropped)

- Full 2.68M-passage NQ and larger-dense runs (bigger-RAM box). The dose-response +
  ~1M-prefix point already establish the trend; these would tighten, not change it.
- Global-corpus RQ3 latency replay (harness re-plumbing).
- Any change to Paper 2 / Component A. This spec is Paper 1 v2 only.

## Success criteria

A self-consistent v2 `main.tex` whose thesis is corpus-dependence, whose headline
numbers are the global-corpus results, whose RQ1–4 are preserved as the per-question
view, and whose headroom claim is quantified under both operationalizations — with
every reported number traceable to a committed artifact.
