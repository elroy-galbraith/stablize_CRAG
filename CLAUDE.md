# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research harness that measures **tool-intent stabilization** on the CRAG
benchmark: how early in a streaming query the retriever's intent "locks in," and
how much tool/retrieval latency can therefore be hidden behind the user's
remaining typing. It produces per-question metrics and the RQ1–RQ4 summary
described in `docs/STUDY.md`.

`docs/PROPOSAL.md` is the authoritative spec — read it for the *why*. It defines the
four research questions (RQ1 descriptive φ distribution; RQ2 streamable fraction
vs. tool latency L and cadence δ; RQ3 validate the bound against the real
pipeline; RQ4 which query features predict early vs. late stabilization) and the
formal metric definitions the code implements verbatim: `t_sc`, `t_suf`, `φ=t*/n`,
volatility `V`, and the hidden-latency bound `H = min(L, max(0, (n−t*)/δ))` (δ in w/s).

The `run_study.py` CLI flags are the proposal's experimental-design factor grid:
`--L` (tool latency), `--delta` (δ, input cadence words/sec), `--top-k`
(sufficiency top-k), `--theta` (coverage threshold θ). Sweep them per the grid in
docs/PROPOSAL.md §8. **Not yet wired** (deliberately, to avoid heavy deps): the dense
retriever condition (H4) and the φ-on-features regression (RQ4 explanatory).

## Critical external dependency

`experiments/crag.py`, `experiments/stabilization.py`, and `experiments/run_study.py`
import from `experiments/streaming_rag.py` (the async streaming harness: `BM25`,
`Config`, `DirectRetrievalBroker`, `run_baseline`, `run_streaming`). It is **vendored**
into this repo (a self-contained, pure-stdlib file) so the study runs without any sibling
checkout. Its source of truth is the separate `streamRAG` project; the file header
documents this. If you re-vendor it, keep the paper-derived `Config` constants
(`query_gen_ms`, `fuse_ms` from Arora et al. Table 3) — they drive the RQ3 latency
validation, and an older harness without them makes RQ3 near-circular (see RQ3 note below).

## Commands

```bash
# Smoke test — no CRAG download needed; writes data/crag_fixture.jsonl.bz2
uv run experiments/make_fixture.py
uv run experiments/run_study.py --data data/crag_fixture.jsonl.bz2 --split 0 --latency-n 5

# Real run (split 0 = validation set); --extra plot enables --plot
uv run --extra plot experiments/run_study.py --data data/crag_task_1_and_2_dev_v4.jsonl.bz2 --split 0 \
  --top-k 3 --L 600 --delta 3 --theta 0.8 --plot
```

Plain `python3 run_study.py …` also works (core is stdlib-only). There is no build
step, no test suite, and no linter configured. The fixture run **is** the smoke
test — use it to verify changes end-to-end.

### Outputs (the reproducible record — inspect these instead of re-running)

- `--out` → `stabilization.csv`: per-question rows. RQ1/RQ2/RQ4 are all recoverable
  from this offline (RQ2 for *any* L/δ/θ, since it's arithmetic over `t_suf`/`t_sc`/`n_words`).
- `--summary-json` → `stabilization.summary.json`: the aggregates (groundable rates,
  RQ1 φ/volatility stats, RQ4 by question_type, RQ2 at the configured point **and the
  full PROPOSAL §8 (L,δ,θ) grid**, plus RQ3 if run), with a `params` block for provenance.
- `--latency-csv` → `latency_validation.csv`: per-question RQ3
  (`measured_saved_ms` vs `H_predicted_ms`) — written only with `--latency-n`. This is
  the **only** result not recoverable from the per-question CSV, so it must be persisted.
- `--plot` → `--plot-out` (`phi_distribution.png`): the φ_suf histogram.

**Canonical artifacts for the paper.** The BM25 split-0, k=3 numbers reported in
`paper/` are sourced from `results/stab_k3.{summary.json,csv}` (central cell
L=600, δ=3, θ=0.8) and the RQ3 replays `results/latency_k3*.csv`. BM25 self-consistency
has tie-break nondeterminism, so an independent rerun shifts `phi_sc` median / volatility
share at the 2nd decimal — re-fill macros from `stab_k3.*`, not from a fresh run.
The dense arm is `results/stabilization.split0.dense.*`; RQ4 inferential stats come
from `experiments/stats.py` → `results/rq4_stats.json`.

Get the CRAG data (CC BY-NC 4.0, research only) — place in `data/`:
```bash
curl -L -o data/crag_task_1_and_2_dev_v4.jsonl.bz2 \
  https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2
```

Dependencies (uv): core has **zero** runtime deps. Optional extras in
`pyproject.toml`: `[html]` (`beautifulsoup4` — cleaner HTML extraction than the
stdlib `HTMLParser` fallback in `html_to_text`), `[plot]` (`matplotlib` — enables
`--plot`), `[all]` (both). The dense-retriever (H4) and RQ4 regression deps are
intentionally not declared yet (see docs/PROPOSAL.md §6/§8).

## RQ3 depends on the harness's latency model

RQ1/RQ2/RQ4 use only BM25 prefix retrieval and are identical regardless of harness
version. **RQ3 is the harness**: it compares the pipeline's measured perceived-latency
saving to the `H = min(L, residual)` bound. With the paper-calibrated `Config`
(`query_gen_ms≈590`, `fuse_ms≈2520`), the pipeline also hides query-generation time,
so measured saving *exceeds* the `H` bound (~1096ms vs 600ms on the fixture) — a real
result: H is conservative. A toy harness where tool latency `L` is the only hideable
cost matches `H` almost exactly, which makes RQ3 circular. If you want `H` to predict
rather than under-predict, fold `query_gen_ms` into the hideable budget in
`hidden_latency_ms` (research-scope, not done).

## Architecture / data flow

The pipeline is a stream of `CragExample`s flowing through three stages:

1. `experiments/crag.py` — `load_crag()` streams the bz2 JSONL, cleans each search
   result's HTML to text (`html_to_text`), chunks into overlapping word-windows
   (`chunk_text`), and grounds the gold answer string(s) into passages to derive
   `gold` (the answer-bearing passage ids, "d*"). `split=0` keeps validation
   rows; `None` keeps all.
2. `experiments/stabilization.py` — `stabilization()` builds a `BM25` index over one
   question's passages, retrieves over every query **prefix** `q[1:t]`, and
   computes `t_sc` (self-consistency), `t_suf` (sufficiency), `phi` = t*/n,
   `volatility`, and the hidden-latency bound `H` (`hidden_latency_ms`).
3. `experiments/run_study.py` — driver. Per-question CSV + the RQ summary. RQ1 = phi/volatility,
   RQ2 = streamable fraction at (L, delta, theta), RQ4 = phi_suf by question_type.
   `--latency-n N` runs RQ3: replays N questions through the async harness
   (`run_baseline` vs `run_streaming`) and compares measured perceived-latency
   savings to the `H` prediction.

## The grounding caveat (must understand before reporting numbers)

CRAG ships **no gold-passage label** — only gold answer strings. `gold_passage_ids`
in `experiments/crag.py` derives d* by normalized string matching: substring match for
multi-token answers, word-boundary match for single-token answers, over `answer`
+ `alt_ans`. Implications baked into the metrics:

- Ungroundable items (false-premise, "I don't know", or answers never present as a
  literal span — common for aggregation/dynamic questions) have empty `gold`, are
  `groundable == False`, and are **excluded from `t_suf`/`phi_suf`**. Always report
  the groundable rate alongside results.
- `t_sc` (self-consistency) needs no grounding and is always defined — report it as
  the grounding-free robustness check.
- `gold_passage_ids` accepts an `llm_judge` fallback for non-verbatim answers, but
  it is **never called by default**. Enabling it changes the groundable population,
  so decide before reporting.

Dense-retriever and regression conditions from the proposal are intentionally not
wired here (they pull in sentence-transformers/torch/stats deps); add them only
for a robustness pass.
