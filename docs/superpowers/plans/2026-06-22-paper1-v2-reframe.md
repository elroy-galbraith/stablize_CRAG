# Paper 1 v2 Reframe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise `paper/main.tex` in place into a v2 whose thesis is that tool-intent stabilization is operationalization-dependent — early stabilization is largely a per-question-pool artifact; under a realistic global corpus it is late, dose-responsive in corpus size, and retriever-general.

**Architecture:** One new stdlib-only analysis script recomputes the streamable fraction under the global corpus from already-committed CSVs. The rest is LaTeX: a new results-macros block, a rewritten front matter and thesis, a new global-results spine section, RQ1–4 demoted to "the per-question view," and a reframed conclusion/limitations. No new experiments; no changes to `experiments/global_corpus.py`, `stabilization.py`, or `crag.py`.

**Tech Stack:** Python 3 stdlib (`csv`, `json`, `argparse`), pytest, LaTeX (`latexmk`/`pdflatex`), the ACL style already in `paper/`.

## Global Constraints

- **In-place v2:** edit `paper/main.tex` and `paper/results.tex`; do not fork a new paper file. One arXiv lineage.
- **Every reported number traces to a committed artifact** under `results/` or an existing v1 macro. One macro per number, with a provenance comment naming its source JSON (match the existing macros block style at `paper/main.tex:18-83`).
- **No new experiments.** Only arithmetic over committed CSVs is allowed (`results/global/**/*.csv`).
- **New code is stdlib-only** (the repo core has zero runtime deps; keep it that way).
- **The headline contrast** is CRAG-native per-question (v1) vs global corpus — NOT the `global_corpus.py` `perq` arm, which is an intermediate control.
- **Delete the v1 claim** that the dense arm shows the effect "is not a BM25 lexical artifact" wherever it appears (abstract + Robustness) — the global dense result reverses its import.
- Spec of record: `docs/superpowers/specs/2026-06-22-paper1-v2-reframe-design.md`.

---

### Task 1: `global_headroom.py` — streamable fraction under the global corpus

**Files:**
- Create: `experiments/global_headroom.py`
- Test: `tests/test_global_headroom.py`

**Interfaces:**
- Consumes: `hidden_latency_ms(t_star:int, n_words:int, L_ms:float, delta_wps:float) -> float` from `experiments/stabilization.py` (pure-stdlib; `H = min(L, max(0,(n-t*))/delta*1000)`).
- Produces:
  - `streamable_fraction(rows: list[dict], L_ms: float, delta_wps: float, theta: float) -> tuple[int, int]` returning `(streamable, denom)`, where `denom` counts rows with a present `t_suf_global` and a row is streamable iff `H >= theta * L_ms`. Mirrors `run_study.py:_streamable_fraction` but uses `t_suf_global` with NO `t_sc` fallback (the global arm has no self-consistency column).
  - `load_rows(csv_path: str) -> list[dict]` returning rows with int `n_words` and `t_suf_global` as `int` or `None` (empty cell → `None`).
  - A `main()` CLI: `--csv` (required), `--L` (default 600), `--delta` (default 3.0), `--theta` (default 0.8), `--label` (default = csv basename), `--out` (summary JSON path). Writes `{"params":{...}, "streamable": s, "denom": d, "streamable_fraction": round(s/d,4)}` and prints one line.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_global_headroom.py
import csv, os
from global_headroom import streamable_fraction, load_rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "n_words", "t_suf_global", "phi_suf_global"])
        w.writeheader()
        w.writerows(rows)


def test_streamable_fraction_basic(tmp_path):
    # row1: n=10,t=2 -> residual=(8/3)*1000=2667ms, H=min(600,2667)=600 >= 0.8*600=480 -> streamable
    # row2: n=10,t=10 -> residual=0, H=0 -> not streamable
    # row3: t_suf_global empty -> excluded from denom entirely
    p = os.path.join(tmp_path, "g.csv")
    _write_csv(p, [
        {"qid": "a", "n_words": 10, "t_suf_global": 2, "phi_suf_global": 0.2},
        {"qid": "b", "n_words": 10, "t_suf_global": 10, "phi_suf_global": 1.0},
        {"qid": "c", "n_words": 10, "t_suf_global": "", "phi_suf_global": ""},
    ])
    rows = load_rows(p)
    s, d = streamable_fraction(rows, L_ms=600, delta_wps=3.0, theta=0.8)
    assert (s, d) == (1, 2)


def test_load_rows_parses_none(tmp_path):
    p = os.path.join(tmp_path, "g.csv")
    _write_csv(p, [{"qid": "c", "n_words": 5, "t_suf_global": "", "phi_suf_global": ""}])
    rows = load_rows(p)
    assert rows[0]["t_suf_global"] is None
    assert rows[0]["n_words"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd experiments && uv run --extra dev python -m pytest ../tests/test_global_headroom.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'global_headroom'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/global_headroom.py
"""Streamable fraction under the global corpus (Paper 1 v2, RQ2 re-analysis).

Arithmetic over the committed global-corpus CSVs (columns:
qid,n_words,t_suf_global,phi_suf_global,t_suf_perq,phi_suf_perq). Reuses the
exact v1 hidden-latency bound so the global number is comparable to run_study's
per-question streamable fraction. No retrieval, no heavy deps.
"""
from __future__ import annotations

import csv
from typing import Optional

from stabilization import hidden_latency_ms


def load_rows(csv_path: str) -> list:
    out = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            t = r.get("t_suf_global")
            out.append({
                "qid": r.get("qid"),
                "n_words": int(r["n_words"]),
                "t_suf_global": int(t) if t not in (None, "") else None,
            })
    return out


def streamable_fraction(rows: list, L_ms: float, delta_wps: float, theta: float):
    streamable = denom = 0
    for r in rows:
        t = r["t_suf_global"]
        if t is None:
            continue
        denom += 1
        if hidden_latency_ms(t, r["n_words"], L_ms, delta_wps) >= theta * L_ms:
            streamable += 1
    return streamable, denom


def main():
    import argparse, json, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--L", type=float, default=600)
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    s, d = streamable_fraction(rows, args.L, args.delta, args.theta)
    frac = round(s / d, 4) if d else None
    label = args.label or os.path.basename(args.csv)
    summary = {"params": {"csv": args.csv, "label": label, "L": args.L,
                          "delta": args.delta, "theta": args.theta},
               "streamable": s, "denom": d, "streamable_fraction": frac}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"{label}: streamable {s}/{d} = {frac}  (L={args.L}, delta={args.delta}, theta={args.theta})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd experiments && uv run --extra dev python -m pytest ../tests/test_global_headroom.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/global_headroom.py tests/test_global_headroom.py
git commit -m "feat: global_headroom.py — streamable fraction under the global corpus"
```

---

### Task 2: Produce the global headroom numbers (run over committed CSVs)

**Files:**
- Create: `results/global/headroom/` (one summary JSON per global CSV)
- Use: `experiments/global_headroom.py` (Task 1), CSVs under `results/global/`

**Interfaces:**
- Consumes: `global_headroom.main()` CLI from Task 1.
- Produces: `results/global/headroom/{label}.summary.json` for each global arm — the `streamable_fraction` numbers Task 3 turns into macros.

- [ ] **Step 1: Run the headroom script over each global CSV**

Run each (from `experiments/`), at the central cell L=600, δ=3, θ=0.8:

This task has two parts: (2a) TDD-enhance `experiments/global_headroom.py` to sweep multiple L values and read either CSV schema; (2b) produce the L-sweep table artifact and run the sanity check.

**Why the change (read first):** the Task-2 sanity gate fired. The binary streamable fraction does NOT collapse at v1's L=600 — NQ/FiQA queries are long enough to hide a small 600 ms tool call, and the gold-guaranteed *subsamples* even exceed v1. The honest result is an **L-sweep**: the global fraction is ≤ per-question at every L and the gap widens with L (collapsing to ≈0.09 at L=2500, the paper's own `fuse_ms`). See spec §7.

**New interfaces (extend Task 1's file):**
- `load_rows(csv_path, t_col="t_suf_global", fallback_col=None) -> list[dict]` — each row `{"qid", "n_words": int, "t_star": int|None}`. `t_star = int(row[t_col])` if that cell is non-empty, else `int(row[fallback_col])` if `fallback_col` is given and non-empty, else `None`. (Renames the per-row key from `t_suf_global` to the schema-neutral `t_star`; update Task 1's test accordingly.)
- `streamable_fraction(rows, L_ms, delta_wps, theta) -> (int,int)` — unchanged logic, now reads `r["t_star"]`.
- `sweep(rows, Ls, delta_wps, theta) -> list[dict]` — `[{"L": L, "streamable": s, "denom": d, "frac": round(s/d,4)}]` for each `L` in `Ls`.

- [ ] **Step 1 (2a): Write the failing tests for the enhancement**

Replace `tests/test_global_headroom.py` with (keeps the basic case, adds dual-schema + sweep; note `t_star` key):

```python
import csv, os
from global_headroom import streamable_fraction, load_rows, sweep


def _write(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)


def test_load_rows_global_schema(tmp_path):
    p = os.path.join(tmp_path, "g.csv")
    _write(p, ["qid", "n_words", "t_suf_global", "phi_suf_global"], [
        {"qid": "a", "n_words": 10, "t_suf_global": 2, "phi_suf_global": 0.2},
        {"qid": "c", "n_words": 5, "t_suf_global": "", "phi_suf_global": ""},
    ])
    rows = load_rows(p)
    assert rows[0]["t_star"] == 2 and rows[0]["n_words"] == 10
    assert rows[1]["t_star"] is None


def test_load_rows_perq_schema_with_fallback(tmp_path):
    # per-question CRAG schema: prefer t_suf, fall back to t_sc
    p = os.path.join(tmp_path, "p.csv")
    _write(p, ["interaction_id", "n_words", "t_sc", "t_suf"], [
        {"interaction_id": "a", "n_words": 8, "t_sc": 3, "t_suf": 5},   # t_suf present -> 5
        {"interaction_id": "b", "n_words": 8, "t_sc": 4, "t_suf": ""},  # falls back to t_sc -> 4
        {"interaction_id": "c", "n_words": 8, "t_sc": "", "t_suf": ""}, # both empty -> None
    ])
    rows = load_rows(p, t_col="t_suf", fallback_col="t_sc")
    assert [r["t_star"] for r in rows] == [5, 4, None]


def test_streamable_fraction_basic(tmp_path):
    # n=10,t=2 -> H=min(600,2667)=600>=480 streamable; n=10,t=10 -> H=0 not; empty -> excluded
    p = os.path.join(tmp_path, "g.csv")
    _write(p, ["qid", "n_words", "t_suf_global"], [
        {"qid": "a", "n_words": 10, "t_suf_global": 2},
        {"qid": "b", "n_words": 10, "t_suf_global": 10},
        {"qid": "c", "n_words": 10, "t_suf_global": ""},
    ])
    s, d = streamable_fraction(load_rows(p), L_ms=600, delta_wps=3.0, theta=0.8)
    assert (s, d) == (1, 2)


def test_sweep_returns_one_cell_per_L(tmp_path):
    p = os.path.join(tmp_path, "g.csv")
    _write(p, ["qid", "n_words", "t_suf_global"], [
        {"qid": "a", "n_words": 12, "t_suf_global": 9},  # residual 3w=1000ms
    ])
    cells = sweep(load_rows(p), [600, 1500, 2500], delta_wps=3.0, theta=0.8)
    assert [c["L"] for c in cells] == [600, 1500, 2500]
    # H=min(L,1000); streamable iff 1000>=0.8L -> L<=1250: true@600, false@1500/2500
    assert cells[0]["frac"] == 1.0 and cells[1]["frac"] == 0.0 and cells[2]["frac"] == 0.0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_global_headroom.py -v`
Expected: FAIL — `load_rows()` got unexpected keyword / `sweep` not defined / `KeyError: 't_star'`.

- [ ] **Step 3: Implement the enhancement**

Edit `experiments/global_headroom.py`:

```python
def load_rows(csv_path: str, t_col: str = "t_suf_global", fallback_col=None) -> list:
    out = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            t = r.get(t_col)
            if t in (None, "") and fallback_col:
                t = r.get(fallback_col)
            out.append({
                "qid": r.get("qid") or r.get("interaction_id"),
                "n_words": int(r["n_words"]),
                "t_star": int(t) if t not in (None, "") else None,
            })
    return out


def streamable_fraction(rows: list, L_ms: float, delta_wps: float, theta: float):
    streamable = denom = 0
    for r in rows:
        if r["t_star"] is None:
            continue
        denom += 1
        if hidden_latency_ms(r["t_star"], r["n_words"], L_ms, delta_wps) >= theta * L_ms:
            streamable += 1
    return streamable, denom


def sweep(rows: list, Ls, delta_wps: float, theta: float) -> list:
    out = []
    for L in Ls:
        s, d = streamable_fraction(rows, L, delta_wps, theta)
        out.append({"L": L, "streamable": s, "denom": d, "frac": round(s / d, 4) if d else None})
    return out
```

Update `main()` to: add `--t-col` (default `t_suf_global`), `--fallback-col` (default `None`), change `--L` to accept a comma list (default `"600,1500,2500"`), call `sweep`, and write `{"params":{...}, "sweep": [...]}` to `--out`. Print one line per L.

- [ ] **Step 4: Run tests, verify pass + full suite**

Run: `uv run --extra dev python -m pytest tests/test_global_headroom.py -v && uv run --extra dev python -m pytest tests/ -q`
Expected: headroom tests pass; no regressions (the 5 `test_global_corpus` failures from missing `bm25s` under `--extra dev` are pre-existing and unrelated).

- [ ] **Step 5: Commit the enhancement**

```bash
git add experiments/global_headroom.py tests/test_global_headroom.py
git commit -m "feat: global_headroom L-sweep + dual-schema (per-question fallback) for the L-dependent headroom table"
```

- [ ] **Step 6 (2b): Produce the L-sweep table artifact**

```bash
cd /home/elroy/projects/stablize_CRAG
uv run python experiments/global_headroom.py --csv results/stab_k3.csv \
  --t-col t_suf --fallback-col t_sc --L 600,1500,2500 --delta 3 --theta 0.8 \
  --label perq_crag --out results/global/headroom/perq_crag.json
uv run python experiments/global_headroom.py --csv results/global/nq_tsuf.1m.csv \
  --L 600,1500,2500 --delta 3 --theta 0.8 --label nq_1m --out results/global/headroom/nq_1m.json
uv run python experiments/global_headroom.py --csv results/global/confirm/fiqa_bm25.csv \
  --L 600,1500,2500 --delta 3 --theta 0.8 --label fiqa_full --out results/global/headroom/fiqa_full.json
```

Expected (must match the spec §7 table): perq_crag 0.739/0.545/0.392; nq_1m 0.586/0.272/0.092; fiqa_full 0.676/0.488/0.308.

- [ ] **Step 7: Sanity check (revised)**

Confirm at EACH L the global fraction is ≤ the per-question fraction, and the per-question−global gap is non-decreasing in L. If any global cell EXCEEDS per-question at the same L, STOP (the gold-guaranteed subsamples do this — they must NOT be used here; only the full corpora above).

- [ ] **Step 8: Commit the artifacts**

```bash
git add results/global/headroom/
git commit -m "results: L-dependent streamable fraction (per-question vs full-corpus global), L in {600,1500,2500}"
```

---

### Task 3: Results-macros block for the global numbers

**Files:**
- Modify: `paper/main.tex:18-83` (the `RESULTS MACROS` block)

**Interfaces:**
- Consumes: JSON summaries in `results/global/confirm/`, `results/global/nq_tsuf.1m.summary.json`, and the L-sweep JSONs in `results/global/headroom/` (Task 2 Step 6).
- Produces: new `\newcommand` macros used by Tasks 5, 8, 9. Names below are the contract; later tasks cite exactly these.

- [ ] **Step 1: Add the global macros**

Insert into the macros block (keep the one-number-per-macro + provenance-comment style). The φ_suf / t_suf=1 values are medians/rates from the named confirm JSONs; the streamable values are from the Task-2 L-sweep JSONs:

```latex
% ---- Global-corpus arm (Paper 1 v2 spine). Sources named per line. ----
% phi_suf dose-response + retriever-generality (gold-guaranteed subsamples)
\newcommand{\phiSufGnqK}{0.571}      % NQ gold+1k median phi_suf; results/global/confirm/nq_bm25_1000.summary.json
\newcommand{\phiSufGnqTenK}{0.625}   % NQ gold+10k median; nq_bm25_10000.summary.json
\newcommand{\phiSufGnqOneM}{0.75}    % NQ ~1M-prefix median; nq_tsuf.1m.summary.json
\newcommand{\phiSufGfiqaBm}{0.636}   % FiQA gold+10k BM25 median; fiqa_bm25_10k.summary.json
\newcommand{\phiSufGfiqaDe}{0.625}   % FiQA gold+10k dense median; fiqa_dense_10k.summary.json
\newcommand{\tSufOneGnqOneM}{0.7\%}  % NQ ~1M t_suf==1 rate; nq_tsuf.1m.summary.json
\newcommand{\tSufOneGfiqaBm}{2.8\%}  % FiQA 10k BM25 t_suf==1; fiqa_bm25_10k.summary.json
\newcommand{\nGfiqaBm}{364}          % FiQA 10k BM25 groundable n; fiqa_bm25_10k.summary.json
\newcommand{\nGfiqaDe}{481}          % FiQA 10k dense groundable n; fiqa_dense_10k.summary.json
% L-dependent headroom: streamable fraction, delta=3, theta=0.8 (full corpora)
% per-question CRAG (v1): results/global/headroom/perq_crag.json
\newcommand{\sfPqSix}{73.9\%}        % L=600  (matches v1 \streamPct)
\newcommand{\sfPqFifteen}{54.5\%}    % L=1500
\newcommand{\sfPqTwentyfive}{39.2\%} % L=2500
% global NQ ~1M: results/global/headroom/nq_1m.json
\newcommand{\sfNqSix}{58.6\%}        % L=600
\newcommand{\sfNqFifteen}{27.2\%}    % L=1500
\newcommand{\sfNqTwentyfive}{9.2\%}  % L=2500
% global FiQA full ~57k: results/global/headroom/fiqa_full.json
\newcommand{\sfFiqaSix}{67.6\%}      % L=600
\newcommand{\sfFiqaFifteen}{48.8\%}  % L=1500
\newcommand{\sfFiqaTwentyfive}{30.8\%} % L=2500
```

- [ ] **Step 2: Verify each macro value against its source file**

For every macro, open the named JSON and confirm the number. Run:

```bash
cd /home/elroy/projects/stablize_CRAG
python3 - <<'PY'
import json
print("nq1k", json.load(open("results/global/confirm/nq_bm25_1000.summary.json"))["dual"]["global"]["phi_suf_median"])
print("nq10k", json.load(open("results/global/confirm/nq_bm25_10000.summary.json"))["dual"]["global"]["phi_suf_median"])
print("nq1m", json.load(open("results/global/nq_tsuf.1m.summary.json"))["dual"]["global"]["phi_suf_median"])
print("fiqaBM", json.load(open("results/global/confirm/fiqa_bm25_10k.summary.json"))["dual"]["global"])
print("fiqaDE", json.load(open("results/global/confirm/fiqa_dense_10k.summary.json"))["dual"]["global"])
PY
```

Expected: prints match the macro values (0.571, 0.625, 0.75, FiQA BM `{phi_suf_median:0.6364, t_suf_eq_1_rate:0.0275, n:364}`, FiQA dense `{phi_suf_median:0.625, n:481, t_suf_eq_1_rate:0.0146}`). Adjust any macro that disagrees (e.g. round 0.6364 → 0.636). Then verify the headroom L-sweep macros against the Task-2 JSONs:

```bash
python3 - <<'PY'
import json
for lab in ("perq_crag","nq_1m","fiqa_full"):
    s=json.load(open(f"results/global/headroom/{lab}.json"))["sweep"]
    print(lab, {c["L"]: c["frac"] for c in s})
PY
```

Expected: `perq_crag {600:0.739,1500:0.545,2500:0.392}`, `nq_1m {600:0.586,1500:0.272,2500:0.092}`, `fiqa_full {600:0.676,1500:0.488,2500:0.308}` — matching `\sfPq*`, `\sfNq*`, `\sfFiqa*` (as percentages).

- [ ] **Step 3: Commit**

```bash
git add paper/main.tex
git commit -m "paper(v2): add global-corpus results macros with provenance"
```

---

### Task 4: Front matter — title + abstract

**Files:**
- Modify: `paper/main.tex:14-15` (title), `paper/main.tex:88-112` (abstract)

**Interfaces:**
- Consumes: macros from Task 3 (`\phiSufGnqOneM`, `\phiSufMed`, `\phiSufGfiqaDe`, `\phiSufGfiqaBm`, `\sfPqTwentyfive`, `\sfNqTwentyfive`, `\streamPct`).

- [ ] **Step 1: Replace the title**

Set a v2 title naming the artifact/cautionary-study framing. Use:

```latex
\title{Tool-Intent Stabilization Is Operationalization-Dependent:\\
A Cautionary Measurement Study of Streaming Retrieval-Augmented Generation}
```

(If the human picks a different title from the alternatives offered at draft time, substitute it here.)

- [ ] **Step 2: Rewrite the abstract around corpus-dependence**

Replace the body of the `abstract` environment (lines 89–111) with prose that makes these claims, in this order, citing the macros:
1. Streaming RAG hides tool latency only when the query stabilizes before input ends — a property we name and measure (tool-intent stabilization), keep.
2. **The measured value depends on the retrieval corpus.** Against a per-question candidate pool (as CRAG ships) stabilization looks early (median $\phi_{\mathrm{suf}}=\phiSufMed$); against a realistic global corpus it is late (median up to $\phiSufGnqOneM$), grows later with corpus size, and holds for both sparse and dense retrieval (FiQA: BM25 $\phiSufGfiqaBm$, dense $\phiSufGfiqaDe$).
3. **Consequence:** the hideable-latency headroom shrinks under the global corpus, and the gap grows with tool latency — at a large tool latency ($L{=}2500$ ms, the regime streaming RAG targets) the streamable fraction falls from $\sfPqTwentyfive$ (per-question) to $\sfNqTwentyfive$ (global NQ). (Do not claim a collapse at small $L$ — at $L{=}600$ long queries still hide it.)
4. The TIS framework ($t_{\mathrm{sc}}$, $t_{\mathrm{suf}}$, $\phi$, the bound $H$) is sound; the caution is to measure TIS against the corpus the deployed system retrieves over.
5. Runs on commodity CPU, needs no training, keep.

**DELETE** the v1 final sentence: "a dense-retriever replication confirms the early-stabilization effect is not a BM25 lexical artifact." Do not paraphrase it — remove the claim.

- [ ] **Step 3: Build to verify no broken macros/refs**

Run: `cd paper && latexmk -pdf -interaction=nonstopmode main.tex >/tmp/v2build.log 2>&1; tail -5 /tmp/v2build.log`
Expected: PDF builds; no `Undefined control sequence` for the new macros. (Pre-existing warnings are fine.)

- [ ] **Step 4: Commit**

```bash
git add paper/main.tex
git commit -m "paper(v2): retitle + rewrite abstract around corpus-dependence; drop the 'not a BM25 artifact' claim"
```

---

### Task 5: Introduction — lead with corpus-dependence

**Files:**
- Modify: `paper/main.tex:114-145` (Introduction)

**Interfaces:**
- Consumes: macros from Task 3.

- [ ] **Step 1: Recast the empirical claim**

Edit the Introduction so the lead empirical statement is corpus-dependence, not "a large early-stabilizing mass with a thin late tail." Concretely:
- Keep the PlayStation example (¶ at lines 126–135) as motivation for *why* the stabilization point matters.
- Replace the "We initially expected ... bimodal ... better described as a large early-stabilizing mass with a thin late tail" sentence (lines 131–135) with: we initially measured stabilization against CRAG's per-question pool and found it early; measuring against a realistic global corpus shows that early picture is largely an artifact of the pool, and stabilization is in fact late — the finding this paper is organized around.
- In the contributions list (lines 137–145), change contribution (i) to frame the bound $H$ as computable but **operationalization-sensitive**, and add a contribution: a measurement caution + the global-vs-per-question evidence (dose-response, two benchmarks, two retrievers).

- [ ] **Step 2: Build + commit**

```bash
cd paper && latexmk -pdf -interaction=nonstopmode main.tex >/tmp/v2build.log 2>&1; tail -3 /tmp/v2build.log
cd .. && git add paper/main.tex && git commit -m "paper(v2): intro leads with corpus-dependence"
```

---

### Task 6: Problem Formalization — make the corpus an explicit parameter of t_suf

**Files:**
- Modify: `paper/main.tex:197-258` (Problem Formalization)

- [ ] **Step 1: Parameterize t_suf by corpus**

In the formalization, define $t_{\mathrm{suf}}$ *with respect to a corpus $C$*: $t_{\mathrm{suf}}(q; C)$ is the first prefix length at which retrieval over $C$ surfaces a gold passage in the top-$k$. State explicitly that v1's results are $t_{\mathrm{suf}}(\cdot; C_{\text{perq}})$ (the per-question pool) and the v2 spine reports $t_{\mathrm{suf}}(\cdot; C_{\text{global}})$. Leave $t_{\mathrm{sc}}$, volatility $V$, and the bound $H$ definitions unchanged. One added paragraph + the $C$ subscript on the $t_{\mathrm{suf}}$ definition is sufficient — do not restructure the section.

- [ ] **Step 2: Build + commit**

```bash
cd paper && latexmk -pdf -interaction=nonstopmode main.tex >/tmp/v2build.log 2>&1; tail -3 /tmp/v2build.log
cd .. && git add paper/main.tex && git commit -m "paper(v2): t_suf defined w.r.t. a retrieval corpus"
```

---

### Task 7: Method — add the global-corpus protocol

**Files:**
- Modify: `paper/main.tex:259-324` (Method)

- [ ] **Step 1: Document the global-corpus protocol**

Add a Method subsection describing, citing `experiments/global_corpus.py`:
- **Gold-guaranteed subsampling:** corpus = all gold passages ∪ $N$ reservoir-sampled distractors, so every query's gold is present; sweep $N$ for the dose-response.
- **Retrievers:** sparse BM25 (`bm25s`, Lucene-style, $k_1{=}1.5,b{=}0.75$) and dense (all-MiniLM-L6-v2, cosine), same prefix protocol.
- **Benchmarks:** BEIR-NQ and BEIR-FiQA (gold qrels supply the gold-passage labels CRAG lacks — note this removes the string-grounding caveat that affects the per-question arm).
- Keep the existing CRAG/per-question method text; present the global protocol as an added measurement condition.

- [ ] **Step 2: Build + commit**

```bash
cd paper && latexmk -pdf -interaction=nonstopmode main.tex >/tmp/v2build.log 2>&1; tail -3 /tmp/v2build.log
cd .. && git add paper/main.tex && git commit -m "paper(v2): method — global-corpus protocol (gold-guaranteed, BM25+dense, NQ+FiQA)"
```

---

### Task 8: Results — new global-corpus spine section + dose-response table

**Files:**
- Modify: `paper/results.tex` (insert a new `\subsection` immediately after `\section{Results}` at line 1, before RQ1 at line 26)

**Interfaces:**
- Consumes: macros from Task 3.

- [ ] **Step 1: Insert the global-vs-per-question results subsection**

Add, as the FIRST results subsection, a `\subsection{Stabilization depends on the retrieval corpus}` containing:
- A `tabular` (booktabs) dose-response table with columns *Operationalization | median $\phi_{\mathrm{suf}}$ | $t_{\mathrm{suf}}{=}1$*, rows: CRAG per-question string-grounded (`\phiSufMed`, `\tSufOneAll`), CRAG per-question precisely-grounded (`\phiSufCleanMed`, `\tSufOneClean`), Global NQ gold+1k (`\phiSufGnqK`), gold+10k (`\phiSufGnqTenK`), ~1M (`\phiSufGnqOneM`, `\tSufOneGnqOneM`), Global FiQA 10k BM25 (`\phiSufGfiqaBm`, `\tSufOneGfiqaBm`), Global FiQA 10k dense (`\phiSufGfiqaDe`).
- Prose stating the three confirmations: (1) **dose-response** — $\phi_{\mathrm{suf}}$ rises monotonically with corpus size on NQ; (2) **two benchmarks** — NQ and FiQA agree; (3) **retriever-general** — on the identical FiQA gold+10k corpus, dense ($\phiSufGfiqaDe$) $\approx$ BM25 ($\phiSufGfiqaBm$); dense surfaces gold for more queries ($\nGfiqaDe$ vs $\nGfiqaBm$) but at the same late point, so it *generalizes* the artifact rather than rescuing the early claim.
- One sentence on the $t_{\mathrm{suf}}{=}1$ collapse: under realistic retrieval one token is almost never sufficient ($\approx$50\% $\to$ $\leq$3\%).
- A one-sentence note that `global_corpus.py`'s top-100 `perq` arm is an intermediate control (later than CRAG's native pool, earlier than full-global), not the headline contrast.

- [ ] **Step 2: Build + commit**

```bash
cd paper && latexmk -pdf -interaction=nonstopmode main.tex >/tmp/v2build.log 2>&1; tail -3 /tmp/v2build.log
cd .. && git add paper/results.tex && git commit -m "paper(v2): new results spine — stabilization depends on the retrieval corpus"
```

---

### Task 9: Demote RQ1–4 to "the per-question view"; quantify the headroom collapse; scope RQ3

**Files:**
- Modify: `paper/results.tex:26` (RQ1 heading + framing), `:113-170` (RQ2), `:171-245` (RQ3), `:246-338` (Robustness)

**Interfaces:**
- Consumes: `\streamPct` and the L-sweep macros `\sfPq*`, `\sfNq*`, `\sfFiqa*` from Task 3.

- [ ] **Step 1: Add the per-question-view framing**

After the new spine subsection, add a one-paragraph bridge: "The remainder of this section reports the *per-question view* — the optimistic operationalization $C_{\text{perq}}$ — which the global results above show over-estimates early stabilization. We retain it because it is the standard CRAG setup and because RQ3 validates the bound mechanism within it." Lightly adjust the RQ1/RQ4 subsection intros to refer to $C_{\text{perq}}$.

- [ ] **Step 2: RQ2 — report the global streamable fraction side-by-side**

In RQ2 (`:113-170`), after the existing `\streamPct` result, add the **L-sweep table** (booktabs), columns *Tool latency $L$ | per-question CRAG | global NQ ~1M | global FiQA ~57k*, rows $L\in\{600,1500,2500\}$ ms, cells from `\sfPq*`/`\sfNq*`/`\sfFiqa*`. Prose: the streamable fraction declines with $L$ for all arms (mechanical); the global fraction is $\le$ per-question at every $L$ and the **gap widens with $L$** — at $L{=}600$ long queries still hide a small tool call ($\sfNqSix$ vs $\sfPqSix$), but at $L{=}2500$ ms (the paper's own calibrated `fuse_ms`$\approx\fuseMs$) the global fraction collapses to $\sfNqTwentyfive$ vs per-question's $\sfPqTwentyfive$. So the artifact bites precisely in the large-tool-latency regime streaming RAG targets. **Do not write a single-number "73.9% → X%" collapse** — it is false at small $L$. Caveats to state: use full corpora (the gold-guaranteed subsamples inflate the fraction and are NOT used here); the global denominator is queries whose gold is retrievable at some prefix; global sufficiency is the hard gold-in-top-$k$ criterion (so $\theta$ is implicit in the global arm — only the streamable threshold $\theta\cdot L$ is shared).

- [ ] **Step 3: RQ3 — scope it explicitly**

In RQ3 (`:171-245`), add one sentence scoping the H-replay as a validation of the bound *mechanism under the per-question (optimistic) operationalization*; note a global-corpus replay is future work (the async harness retrieves over per-question passages; re-plumbing to a 10k+ global index is out of scope). Do not change the RQ3 numbers.

- [ ] **Step 4: Robustness — remove the reversed dense claim**

In Robustness (`:246-338`), find the dense-retriever paragraph and **remove/rewrite** any statement that the dense arm shows the early effect "is not a BM25 lexical artifact." Replace with: the per-question dense arm reproduces the per-question early pattern, but the *global* dense arm (Task 8) shows the artifact generalizes to dense retrieval — cross-reference the spine subsection.

- [ ] **Step 5: Build + commit**

```bash
cd paper && latexmk -pdf -interaction=nonstopmode main.tex >/tmp/v2build.log 2>&1; tail -3 /tmp/v2build.log
cd .. && git add paper/results.tex && git commit -m "paper(v2): demote RQ1-4 to per-question view; global headroom collapse; scope RQ3; fix dense claim"
```

---

### Task 10: Conclusion + Limitations

**Files:**
- Modify: `paper/main.tex:328-361` (Conclusion), `:362-end` (Limitations)

- [ ] **Step 1: Reframe the conclusion**

Lead the Conclusion with the measurement caution: TIS is real and useful but operationalization-dependent; report it against the corpus the deployed system retrieves over. Keep the TIS framework as the durable contribution. Remove any surviving "stabilization is often early" headline phrasing.

- [ ] **Step 2: Update Limitations**

Add: (a) the per-question candidate pool is the source of the v1 over-estimate; (b) the global corpus is itself a BEIR construction (NQ/FiQA qrels), not the deployed system's corpus, so the true operating number is system-specific; (c) full-2.68M-passage and larger-dense runs are deferred (the dose-response + ~1M point establish the trend). Keep the existing grounding-precision limitation.

- [ ] **Step 3: Build + commit**

```bash
cd paper && latexmk -pdf -interaction=nonstopmode main.tex >/tmp/v2build.log 2>&1; tail -3 /tmp/v2build.log
cd .. && git add paper/main.tex && git commit -m "paper(v2): conclusion + limitations reframed around the measurement caution"
```

---

### Task 11: Final verification — build clean + claim traceability

**Files:**
- Read-only pass over `paper/main.tex`, `paper/results.tex`

- [ ] **Step 1: Clean build, no undefined refs/macros**

Run: `cd paper && latexmk -C >/dev/null 2>&1; latexmk -pdf -interaction=nonstopmode main.tex >/tmp/v2final.log 2>&1; grep -iE 'undefined|error|\\?\\?' /tmp/v2final.log | grep -viE 'hyperref|rerun' | head`
Expected: no `Undefined control sequence`, no `LaTeX Error`, no `??` cross-refs. PDF exists at `paper/main.pdf`.

- [ ] **Step 2: Claim-traceability pass**

Read the abstract, intro, the new spine subsection, and the conclusion. For every quantitative claim, confirm the macro it cites exists in the Task 3 block and the macro's provenance comment names a real committed file. List any claim with no backing number; if found, fix the prose or add the macro. Confirm the deleted "not a BM25 artifact" claim appears NOWHERE (grep):

```bash
grep -niE 'not a .*(bm25|lexical) artifact|is not a bm25' paper/*.tex
```
Expected: no matches.

- [ ] **Step 3: Run the full test suite (no regressions)**

Run: `cd experiments && uv run --extra dev --extra global --extra dense python -m pytest ../tests/ -q`
Expected: all pass (includes the Task 1 headroom tests and the existing global_corpus tests).

- [ ] **Step 4: Final commit (if the traceability pass changed anything)**

```bash
git add -A && git commit -m "paper(v2): final verification — traceability + clean build" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- Title + abstract (spec §1) → Task 4. ✓
- Intro (spec §2) → Task 5. ✓
- Corpus-parameterized t_suf (spec §3) → Task 6. ✓
- Method / global protocol (spec §4) → Task 7. ✓
- New global-results spine (spec §5) → Task 8. ✓
- RQ1–4 demotion (spec §6) → Task 9 Step 1. ✓
- Headroom recompute + RQ3 scope (spec §7) → Task 1+2 (compute), Task 9 Steps 2–3. ✓
- Conclusion + Limitations (spec §8) → Task 10. ✓
- `global_headroom.py` component (spec Components) → Task 1. ✓
- Macros (spec Components) → Task 3. ✓
- Delete the dense "not a BM25 artifact" claim (spec Global Constraint) → Task 4 Step 2, Task 9 Step 4, Task 11 Step 2 grep. ✓
- Out-of-scope items (full 2.68M, global RQ3 replay, Paper 2) → named in Task 10 Limitations + Task 9 Step 3. ✓

**Placeholder scan:** no `<fill>` remain — Task 3's headroom macros carry concrete verified values (`\sfPq*`/`\sfNq*`/`\sfFiqa*`) produced and checked in Task 2. Every code step shows complete code. LaTeX prose steps specify the exact claims, the macros to cite, and the sentences to delete.

**Type consistency:** `streamable_fraction(rows, L_ms, delta_wps, theta) -> (int,int)`, `load_rows(path, t_col, fallback_col) -> list[dict]` (rows keyed by `t_star`), and `sweep(rows, Ls, delta_wps, theta) -> list[dict]` are defined in Tasks 1–2 and consumed identically. Macro names defined in Task 3 (`\phiSufGnqOneM`, `\sfPqTwentyfive`, `\sfNqTwentyfive`, `\sfFiqaTwentyfive`, `\phiSufGfiqaBm`, `\phiSufGfiqaDe`, `\nGfiqaDe`, `\nGfiqaBm`, etc.) are the exact names cited in Tasks 4, 5, 8, 9. `hidden_latency_ms` signature matches `experiments/stabilization.py:100`.

**Headroom-framing revision (post-Task-2 sanity gate):** Task 2 is now an L-sweep (not a single L=600 number); the streamable fraction does not collapse at small L (long queries mask it) but the global−per-question gap widens with L. Spec §7 and Tasks 2/3/4/9 are aligned to this. User approved "L-dependent headroom."
