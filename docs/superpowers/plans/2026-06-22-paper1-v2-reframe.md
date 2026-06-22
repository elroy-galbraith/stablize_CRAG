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

```bash
cd experiments
for pair in \
  "nq_bm25_1000:../results/global/confirm/nq_bm25_1000.csv" \
  "nq_bm25_10000:../results/global/confirm/nq_bm25_10000.csv" \
  "nq_1m:../results/global/nq_tsuf.1m.csv" \
  "fiqa_bm25_10k:../results/global/confirm/fiqa_bm25_10k.csv" \
  "fiqa_dense_10k:../results/global/confirm/fiqa_dense_10k.csv"; do
  label="${pair%%:*}"; csv="${pair##*:}"
  uv run python global_headroom.py --csv "$csv" --L 600 --delta 3 --theta 0.8 \
    --label "$label" --out "../results/global/headroom/${label}.summary.json"
done
```

Expected: five printed lines like `nq_1m: streamable 1373/... = 0.xx`. Record each `streamable_fraction`.

- [ ] **Step 2: Sanity-check against the per-question baseline**

The v1 per-question streamable fraction is `\streamPct{}` = 73.9%. Confirm the global fractions are **lower** (the headroom collapse the spec predicts). If any global fraction is ≥ 73.9%, STOP and re-examine — that would contradict the thesis and must be understood before writing prose.

- [ ] **Step 3: Commit the headroom artifacts**

```bash
git add results/global/headroom/
git commit -m "results: global-corpus streamable fraction (headroom collapse) at L=600,d=3,theta=0.8"
```

---

### Task 3: Results-macros block for the global numbers

**Files:**
- Modify: `paper/main.tex:18-83` (the `RESULTS MACROS` block)

**Interfaces:**
- Consumes: the JSON summaries in `results/global/confirm/`, `results/global/nq_tsuf.1m.summary.json`, and `results/global/headroom/` (Task 2).
- Produces: new `\newcommand` macros used by Tasks 5–6. Names below are the contract; later tasks cite exactly these.

- [ ] **Step 1: Add the global macros**

Insert into the macros block (keep the one-number-per-macro + provenance-comment style). Fill the streamable values from Task 2's outputs (shown as `<...>`); the φ_suf / t_suf=1 values are the medians/rates already in the named JSON files:

```latex
% ---- Global-corpus arm (Paper 1 v2 spine). Sources named per line. ----
\newcommand{\phiSufGnqK}{0.571}      % NQ gold+1k median phi_suf; results/global/confirm/nq_bm25_1000.summary.json
\newcommand{\phiSufGnqTenK}{0.625}   % NQ gold+10k median; nq_bm25_10000.summary.json
\newcommand{\phiSufGnqOneM}{0.75}    % NQ ~1M-prefix median; nq_tsuf.1m.summary.json
\newcommand{\phiSufGfiqaBm}{0.636}   % FiQA gold+10k BM25 median; fiqa_bm25_10k.summary.json
\newcommand{\phiSufGfiqaDe}{0.625}   % FiQA gold+10k dense median; fiqa_dense_10k.summary.json
\newcommand{\tSufOneGnqOneM}{0.7\%}  % NQ ~1M t_suf==1 rate; nq_tsuf.1m.summary.json
\newcommand{\tSufOneGfiqaBm}{2.8\%}  % FiQA 10k BM25 t_suf==1; fiqa_bm25_10k.summary.json
\newcommand{\nGfiqaBm}{364}          % FiQA 10k BM25 groundable n; fiqa_bm25_10k.summary.json
\newcommand{\nGfiqaDe}{481}          % FiQA 10k dense groundable n; fiqa_dense_10k.summary.json
\newcommand{\streamPctGnqOneM}{<fill>\%}   % global streamable frac, NQ ~1M; results/global/headroom/nq_1m.summary.json
\newcommand{\streamPctGfiqaBm}{<fill>\%}   % global streamable frac, FiQA 10k BM25; headroom/fiqa_bm25_10k.summary.json
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

Expected: prints match the macro values (0.571, 0.625, 0.75, FiQA BM `{phi_suf_median:0.6364, t_suf_eq_1_rate:0.0275, n:364}`, FiQA dense `{phi_suf_median:0.625, n:481, t_suf_eq_1_rate:0.0146}`). Adjust any macro that disagrees (e.g. round 0.6364 → 0.636).

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
- Consumes: macros from Task 3 (`\phiSufGnqOneM`, `\phiSufMed`, `\phiSufGfiqaDe`, `\phiSufGfiqaBm`, `\streamPctGnqOneM`, `\streamPct`).

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
3. **Consequence:** the hideable-latency headroom largely collapses — the per-question streamable fraction $\streamPct$ drops to $\streamPctGnqOneM$ under the global corpus.
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
- Consumes: `\streamPct`, `\streamPctGnqOneM`, `\streamPctGfiqaBm` from Task 3.

- [ ] **Step 1: Add the per-question-view framing**

After the new spine subsection, add a one-paragraph bridge: "The remainder of this section reports the *per-question view* — the optimistic operationalization $C_{\text{perq}}$ — which the global results above show over-estimates early stabilization. We retain it because it is the standard CRAG setup and because RQ3 validates the bound mechanism within it." Lightly adjust the RQ1/RQ4 subsection intros to refer to $C_{\text{perq}}$.

- [ ] **Step 2: RQ2 — report the global streamable fraction side-by-side**

In RQ2 (`:113-170`), after the existing `\streamPct` result, add a sentence + (optionally) two table rows: under the global corpus the streamable fraction falls to `\streamPctGnqOneM` (NQ ~1M) / `\streamPctGfiqaBm` (FiQA 10k BM25), at the same $L{=}600$, $\delta{=}3$, $\theta{=}0.8$. State the comparison caveat: the global denominator is queries whose gold is retrievable at some prefix, and global sufficiency is the hard gold-in-top-$k$ criterion (so $\theta$ is implicit in the global arm; only the streamable threshold $\theta\cdot L$ is shared).

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

**Placeholder scan:** `<fill>` macros in Task 3 are intentional — they are the deliverable of Task 2, with the exact source file named and a verification step. Every code step shows complete code. LaTeX prose steps specify the exact claims, the macros to cite, and the sentences to delete.

**Type consistency:** `streamable_fraction(rows, L_ms, delta_wps, theta) -> (int,int)` and `load_rows(path) -> list[dict]` are defined in Task 1 and consumed identically in Tasks 1–2. Macro names defined in Task 3 (`\phiSufGnqOneM`, `\streamPctGnqOneM`, `\phiSufGfiqaBm`, `\phiSufGfiqaDe`, `\nGfiqaDe`, `\nGfiqaBm`, etc.) are the exact names cited in Tasks 4, 5, 8, 9. `hidden_latency_ms` signature matches `experiments/stabilization.py:100`.
