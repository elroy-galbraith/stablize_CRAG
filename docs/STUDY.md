# RQ1–RQ3 study glue (CRAG → stabilization → harness)

Measures **tool-intent stabilization** on CRAG and validates the latency bound
against the streaming harness. Files:

- `crag.py` — load CRAG (bz2 JSONL), clean HTML → text, chunk into passages,
  ground gold answers to passage ids (derives d*).
- `stabilization.py` — per-question t_sc, t_suf, φ, volatility, hidden-latency bound H.
- `run_study.py` — driver: per-question CSV + RQ1/RQ4 summary + RQ2 streamable
  fraction, optional RQ3 latency validation.
- `make_fixture.py` — synthetic CRAG-format file for testing without the download.
- `streaming_rag.py` — the streaming harness, **vendored** into this repo (source
  of truth is the `streamRAG` project). Keep its paper-derived `Config` latency
  constants for RQ3; see the file header and CLAUDE.md.

## Get the data

CRAG is CC BY-NC 4.0 (research only). Task 1 & 2 dev set:

```
curl -L -o crag_task_1_and_2_dev_v4.jsonl.bz2 \
  https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2
```

## Run

```
# smoke test (no download needed)
python3 make_fixture.py
python3 run_study.py --data crag_fixture.jsonl.bz2 --split 0 --latency-n 5

# real run (split 0 = validation)
python3 run_study.py --data crag_task_1_and_2_dev_v4.jsonl.bz2 --split 0 \
  --top-k 3 --L 600 --delta 3 --theta 0.8 --out stabilization.csv --plot
```

Each run persists three structured artifacts for the report (no need to re-run to
re-read results): `stabilization.csv` (per-question), `stabilization.summary.json`
(RQ1/RQ4 stats + the RQ2 (L,δ,θ) grid + RQ3 means + a `params` provenance block),
and — with `--latency-n` — `latency_validation.csv` (per-question RQ3 measured-vs-H,
the only output not recomputable from the per-question CSV). Override paths with
`--summary-json` / `--latency-csv` / `--plot-out`.

## Write-up

`paper/` holds the LaTeX write-up (`main.tex` + `results.tex` + `refs.bib`); build
with `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
`make_figures.py` (run `uv run --extra plot python3 make_figures.py`) rebuilds all
paper figures from `results/*` — it reads the saved CSVs/JSON, so it needs no CRAG
data and no re-run. The committed results in `paper/` come from the full split-0
run (k∈{1,3,5}, RQ3 on 60 questions); see `results/run_k*.log`.

## Dependencies

Core run needs nothing beyond the Python stdlib + the harness. Optional:
`beautifulsoup4` (cleaner HTML than the stdlib fallback), `matplotlib` (`--plot`).
Dense-retriever and regression conditions from the proposal are not wired here
(deliberately — they pull in `sentence-transformers`/`torch` and a stats lib);
add them only for the robustness pass.

## The grounding caveat (read this)

CRAG has no gold-passage label, only gold answer strings. `gold_passage_ids`
derives d* by normalized string match (substring for multi-token answers,
word-boundary for single-token), using `answer` + `alt_ans`. Consequences you
must report:

- **Groundable rate** — share of questions whose answer is findable in the pages.
  Ungroundable items (false-premise, "I don't know", and answers not present as a
  string — common for aggregation/dynamic questions) are excluded from t_suf.
- An `llm_judge` hook exists in `gold_passage_ids` as a fallback for answers that
  aren't verbatim spans; it is never called by default. Decide whether to enable
  it before you report numbers — it changes the groundable population.

t_sc (self-consistency) needs no grounding and is always defined, so report it
alongside t_suf as a grounding-free robustness check.
