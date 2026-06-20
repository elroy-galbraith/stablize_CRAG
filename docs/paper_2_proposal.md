# Research Plan — Paper 2: A Streaming Tool-Use Trigger

## From the tool-intent stabilization *measure* to a live *system*

**Project:** stablize_CRAG **Paper 1 status:** Complete, on arXiv — TIS as an intrinsic *measurement* (CRAG; BM25 **and** dense). Targeting EACL 2027 (Findings-eligible) via the Aug 2026 ARR cycle. **This document:** Spec for **Paper 2**, a *system* paper that turns the φ_suf measure into a live retrieval trigger and evaluates it end-to-end on latency, compute, and generated-answer quality. **Target:** Main-venue long paper (EMNLP or ACL). Must cite, and post-date, Paper 1. **Core new contribution:** A lightweight, model-agnostic classifier that decides *when* to fire speculative retrieval in a live stream — trained on the φ_suf labels Paper 1 produces — plus an end-to-end demonstration that firing early saves latency and compute **without degrading answer quality**, and a cross-retriever TIS × quality analysis that gives the metric external validity.

---

## Why this is a second paper, not an extension

This was a deliberate decision (see also `PROPOSAL.md` for Paper 1). The seam is clean enough that reviewers should not read it as salami-slicing — but the defense has to survive a reviewer who has *read Paper 1*, so it is stated carefully below.

- **Paper 1 — retrospective, intrinsic measurement.** Defines TIS; characterizes the φ_sc / φ_suf distribution on CRAG under BM25 *and* dense; derives and validates the latency-hiding bound `H`; shows early stabilization is not a BM25 lexical artifact. It stops at *retrieving the gold document* — no generation, no live trigger.
- **Paper 2 — prospective system + downstream quality (this spec).** A live classifier trigger; end-to-end answer generation; resource/latency evaluation; a cross-retriever quality table; a second benchmark. It *cites* Paper 1 for the metric and the distribution and re-reports none of it as novel.

**One-line test for every Paper 2 contribution:** "Could this have gone in Paper 1?" If yes, it does not belong here. The dense-retriever results are the canonical example — they are *in Paper 1*, so Paper 2 treats them as established, not as a phase.

**The two seams that need active defense.** The arXiv Paper 1 (`paper/main.tex`) names two of Paper 2's pieces as its *own* explicit next steps: its Conclusion lists "regress $\phi$ on query features," and its Limitations calls cross-dataset / cross-lingual replication "the natural next step, not optional polish." Crucially — and verified against the submitted text — Paper 1 reports **neither as a result**: the only feature analysis it ships is the question-type Kruskal–Wallis/Dunn test (`experiments/stats.py`; the regression was deliberately never wired, see `CLAUDE.md`), and it runs a single English benchmark. So the seam is sound on the *binding* question — what Paper 1 actually claims — but Paper 1 *advertises* both as its roadmap, which raises, not lowers, the differentiation bar. The defense:

1. **Differentiate in kind, not just in timing.** Paper 2 must not ship a standalone φ-on-features regression or a standalone "does φ replicate on benchmark 2?" measurement — those *are* the Paper-1 next steps and would read as phase 2. Component A's feature work is the *trigger's* ablation: which prefix-observable, **live** signal the classifier fires on — a question that exists only once there is a live commit decision, which Paper 1 never builds. Component E tests whether the *trigger's latency/quality wins* transfer, not whether φ does.
2. **Reframe both as serving the trigger, never as standalone.** Component A's feature-importance ablation exists to adjudicate *which prefix-observable signal the live trigger fires on* (retrieval-stability vs. entity-position) — a question only a live system raises. Component E exists to test whether *the trigger's latency/quality wins* generalize, not to re-measure φ. Neither is a free-standing analysis contribution.

---

## What Paper 1 established (the foundation Paper 2 stands on)

Headline numbers from `results/` (CRAG split-0; canonical cell L=600 ms, δ=3 w/s, θ=0.8):

| Quantity | BM25 (k=3) | Dense | Reading |
| :---- | :---- | :---- | :---- |
| φ_suf median (sufficiency point / n) | **0.143** | **0.222** | Most queries are answerable *well* before the user stops typing |
| φ_sc median (self-consistency) | 0.727 | 0.933 | Grounding-free robustness check |
| Retrieved-gold population | 292 / 1371 (21%) | 363 / 1371 (27%) | Dense finds gold more often |
| Groundable rate | 485 / 1371 (35%) | 485 / 1371 (35%) | t_suf is only defined on this subset (caveat below) |
| Streamable fraction @ (600, 3, 0.8) | 0.74 | 0.54 | Share of queries admitting latency hiding |

Five facts that directly shape Paper 2:

1. **There is large headroom for a trigger.** φ_suf is low under both retrievers — the answer-bearing document is usually retrievable from a short prefix. A system that fires early has a lot to gain.
2. **Early stabilization is a query property, not lexical matching.** It survives the switch to dense retrieval (abstract: "a dense-retriever replication confirms the early-stabilization effect is not a BM25 lexical artifact"). The trigger can therefore be built on a retriever-agnostic signal.
3. **`H` is a conservative *aggregate floor*, not a per-query predictor — and the per-query failures are real, not hypothetical.** On the BM25 replay, realized savings *exceed* `H` on average (≈1122 ms measured vs. 590 ms predicted, n=60). But the dense replay tells the other half of the story: **20% of queries (12/60) realize *net-negative* perceived savings** (`results/stabilization.split0.dense.summary.json`), with mean measured saving 730 ms vs. `H` 586 ms. The same BM25 replay goes net-negative on only **1/60 (≈1.7%)** — the rate Paper 1 actually reports — so the dense failure rate is a **≈12× jump that Paper 1 never reports at all** (the paper's dense arm covers only retrieved-gold rate, φ_sc, φ_suf, and the by-type ranking). `H` is an aggregate floor that the mean clears while a sizeable per-query tail goes negative. → This 1-in-5 net-negative rate is exactly the gap a *per-query* trigger fills, and it is Paper 2's sharpest motivation (Component D).
4. **Question type is a weak predictor of φ_suf.** RQ4 (BM25, `results/rq4_stats.json`): Kruskal–Wallis H = 17.0, p = .017, ε² ≈ 0.035–0.058; on the k=5 large classes, p = .0075, ε² ≈ 0.037–0.051; **no** Dunn pair survives Holm correction (closest, simple vs. aggregation, p_holm = .051). → A fixed question-type heuristic will not work. This *motivates* a learned trigger on richer prefix signals — the weak effect is the reason the classifier earns its keep, not a problem.
5. **Better retrieval quality came with *worse* streamability.** Dense retrieves gold more often than BM25 (363 vs. 292) yet stabilizes *later* (φ_suf 0.222 > 0.143) and has a lower streamable fraction (0.54 vs. 0.74). The one cross-retriever comparison Paper 1 already has shows quality and TIS moving the "wrong" way together. This directly constrains Component C's thesis (see the caveat there) and must be confronted, not assumed away.

**Grounding caveat (carries over, and now bites the trigger).** Only ~35% of CRAG items are groundable for t_suf (gold answer string locatable as a span). t_sc is always defined. Paper 2 must report on both populations — and crucially, **the trigger is trained only on the *retrieved-gold* subset** where t_suf is actually defined (292/1371 = **21%**, *narrower even than* the 35% groundable set — 193 groundable items still lack a label because BM25 never surfaces gold), yet deployed on all queries (see Component A, "the population problem"). The end-to-end generation eval (Component B) is what reaches the ungroundable items — it does not need a gold-passage label.

### Paper 1 (done) vs. Paper 2 (this spec)

| Axis | Paper 1 | Paper 2 |
| :---- | :---- | :---- |
| Retriever | BM25 + dense | reuse both; add a reranker as a 3rd point on the TIS × quality curve |
| Output | gold-document retrieval | **generated answer** + quality score |
| Trigger | none (and a fixed-interval baseline) | **learned classifier** trigger |
| Evaluation | intrinsic (φ, H) | **end-to-end**: perceived latency + compute + answer quality |
| RQ3 replay | n = 60 | n ≥ 200, bootstrap CIs |
| Benchmark | CRAG | CRAG + NQ or TriviaQA |
| Training | none | classifier trained offline on split 1; **training-free at inference** |

### Positioning vs. Stream RAG (carry forward, tightened)

Arora et al. fine-tune the LLM itself to decide when to fire, trained on downstream QA-accuracy signals — entangling retriever quality, model quality, and query difficulty, and requiring LLM fine-tuning. Our trigger is trained on **φ_suf** — retrieval sufficiency at the prefix level, a query-intrinsic signal computable without the LLM. Stream RAG never defines this target. Position our trigger as the principled, lightweight, CPU-only foundation that explains *why* a trigger should fire.

**Be honest about the comparison we can actually run.** A reproduction of Stream RAG's LLM trigger is not on the critical path (it requires their fine-tuning pipeline). We therefore make the *principled/lightweight/interpretable* claim — model-agnostic, CPU-only, training-free at inference — and substantiate it against the **fixed-interval baseline we control**. We do **not** claim to beat Stream RAG on accuracy unless we actually reproduce it; if a usable reference implementation surfaces we add it as a point of comparison, but the contribution does not rest on out-scoring a heavier learned system.

---

## Component A — The classifier trigger (core)

The paper measures t_suf *retrospectively*. The classifier predicts, at each word position *t* in a live stream, whether sufficiency has probably been reached, so the trigger fires as soon as it is confident rather than on a fixed interval.

### The observability problem (and why it is not fatal)

φ_suf = t_suf / n. Live, neither *n* (total length) nor *d\** (gold document) is known, so `t ≥ t_suf` cannot be observed at inference. But this is the *standard* learned-commit setup: the **label is retrospective; only the features must be live-observable**. It mirrors simultaneous-MT commit classifiers ("enough source seen?") and incremental SLU (intent before the utterance ends). The classifier learns:

> observable prefix signals at *t* → P(t ≥ t_suf), fire when P exceeds a threshold.

Note the classifier never optimizes perceived-latency saving directly — *saving* depends on *n* and is computable only retrospectively, so it is a downstream **evaluation** quantity, not a training signal or a feature. The model optimizes the live-observable proxy P(t ≥ t_suf); saving is what we measure afterward.

**The Reflector bounds the downside.** If the trigger fires at *t* and retrieval misses *d\**, the Reflector re-fires later. A false positive costs one wasted retrieval call, **not a wrong answer**. This asymmetry shapes the loss — but it trades against compute (see "the competing-objectives problem" below), so the weighting is an empirical choice, not a free "fire early" license.

### The population problem (the hole a methods reviewer will name)

`t_suf` is defined only on the **retrieved-gold** subset (292/1371 = 21% under BM25): the gold answer must be both groundable *and* actually surfaced in some prefix's top-k. Note this is **narrower than "groundable" (485 = 35%)** — 193 groundable queries never have BM25 retrieve their gold, so they too lack a `t_suf` label. The per-word label `1[t ≥ t_suf]` therefore *cannot exist* for the ~79% of queries where the gold document is never surfaced from a prefix — there, `t_suf = ∞`. Two consequences must be handled explicitly or the trigger is trained on one population and deployed on another:

- **Training population.** The classifier's supervised signal exists only on the 21% retrieved-gold set — not the 35% groundable set, and certainly not the full workload. State the 21% plainly; it is the honest scope of the t_suf signal.
- **Deployment behavior on ungroundable queries.** At inference we cannot tell groundable from ungroundable. So Paper 2 reports trigger behavior **separately** on the two populations on the test split, and adds a **secondary, always-defined target** built on `t_sc` (self-consistency, which needs no gold label) so that "has the top-1 settled?" is learnable for *every* query. The headline trigger is the t_suf model; the t_sc model is the safety check that the trigger does not fire pathologically (e.g., constantly, or never) on the ungroundable majority. If the two targets disagree sharply on the ungroundable set, that is itself a reportable limit on how far a sufficiency-trained trigger generalizes.

### The competing-objectives problem (latency vs. compute pull apart)

Paper 2 sells **two** wins: perceived-latency saving *and* a reduced retrieval-call count (the classifier fires ~once near t_suf, vs. fixed-interval's many speculative fires). These objectives conflict. The "late-fire is worse than early-fire" asymmetry pushes the threshold *down* to avoid lost latency — but every premature fire is an extra speculative retrieval, eroding the compute advantage. We therefore:

- treat the firing threshold (and the loss class-weight) as a **single knob** that traces a **latency ↔ compute Pareto frontier**, and report the frontier, not a single operating point claimed to maximize both;
- pick the headline operating point by a stated rule (e.g., the knee of the frontier, or "match fixed-interval's mis-fire rate, beat it on mean saving"), so the comparison to fixed-interval is apples-to-apples on *one* axis at a time.

### Features (≤ 6; all live-observable)

Paper 1's RQ4 result reorders the priors here. Because question type explains very little φ_suf variance, lead with the **retrieval-stability** signal and absolute word count; treat entity position as a hypothesis the ablation tests, not as an assumed dominant cause.

**Retrieval-side** (run the prefix retriever on each `q[1:t]`):

| Feature | Rationale |
| :---- | :---- |
| `top1_stable_streak` | consecutive prefixes with the same top-1 — a t_sc proxy, observable without *d\** (likely the dominant signal) |
| `top1_changed_since_last` | did the top-1 change vs. *t*−1? volatility ⇒ not yet stable |

**Query-side** (token stream only, no retrieval):

| Feature | Rationale |
| :---- | :---- |
| `t` (absolute word count) | more words ⇒ higher P(t ≥ t_suf). Use absolute *t*, **never** t/n — *n* is unknown live |
| `named_entity_detected` | a topic anchor is present — directly observable |
| `entity_word_position` | position of the first NE — early anchor ⇒ early stabilization (hypothesis, tested by ablation) |
| `question_word_type` | who/what/when/where/which — a *weak* prior, given RQ4 |

Model: logistic regression or a small GBDT — interpretable, CPU-only, trains in seconds. The **feature-importance ablation is a headline result**: it adjudicates retrieval-stability vs. entity-position as the operative signal, and it lets Paper 2 *directly measure* first-NE word position against t_suf — closing the "entity-position account" that Paper 1 could only offer as interpretation.

### Training & evaluation

- **Labels:** per-word binary, 1 if `t ≥ t_suf`. **These do not exist yet for the training split.** Paper 1's per-question CSVs are split-0 only; training on split 1 requires first running the Paper-1 stabilization pipeline on split 1 (1,335 questions — confirmed present in `crag_task_1_and_2_dev_v4.jsonl.bz2`) to generate its t_suf labels. This is cheap (CPU, hours) and yields a **clean train(split 1) / test(split 0) separation with no leakage** — a strength to state, not hide.
- **Label-quality audit (required for a system paper).** t_suf rests on substring/word-boundary grounding (the `llm_judge` fallback is off by default), so the training labels are noisy and that noise caps the classifier's achievable accuracy. Sample N≈100 groundable items, adjudicate the gold-passage grounding by hand (or LLM-judge with spot human checks), and report label precision. Without this, reviewers cannot tell trigger error from label error.
- **Train:** CRAG split 1 (1,335 questions; the *retrieved-gold* subset — ~21% by split-0 rates — carries t_suf labels, not the full or groundable set). **Test:** the expanded replay set (Component D), drawn from split 0.
- **Loss:** weighted BCE, class-weight set on the latency↔compute frontier (above), not a fixed "upweight positives" rule.
- **Baselines:** fixed-interval trigger (Paper 1's) and, **only if a usable implementation is available**, Stream RAG's LLM trigger.

| Metric | Captures |
| :---- | :---- |
| Mean perceived-latency saving (ms) | upside |
| Net-negative / mis-fire rate (%) | downside (Reflector-correctable) |
| Retrieval-call count per query | compute cost (the other axis of the Pareto frontier) |
| Median saving at correct fires | quality of the hits |
| Spearman ρ (fire-*t* vs. true t_suf) | per-query ranking |
| Feature importance (ablation) | which signal the model actually uses |

**Target result:** the classifier fires earlier on early-stabilizing queries and later on late ones, beating fixed-interval on mean saving while matching or beating it on mis-fire rate **at equal or lower retrieval-call count**.

---

## Component B — End-to-end generation & answer quality (new; answers Jordan)

Paper 1 stops at the gold document. Paper 2 closes the loop: feed the *triggered* retrieval context to a **fixed** generator and produce an answer.

- **Headline claim to test:** firing at the classifier's point preserves answer quality relative to a non-speculative full-query baseline (and vs. fixed-interval), with the Reflector recovering mis-fires.
- **Metrics:** CRAG-style accuracy / truthfulness (accurate / missing / incorrect, with a hallucination penalty) as primary; EM/F1 as a cheap proxy; report hallucination rate explicitly.
- **Control:** hold the generator fixed across all trigger conditions, so the comparison isolates *when to fire*, not model quality. Do **not** sweep generators.
- **Stated scope of the quality claim.** Because the generator is fixed to one small open instruct model, "preserves answer quality" is established *for that capability tier*. A stronger generator may be more (or less) robust to partial/early context; we flag the result as generator-conditional and name testing a frontier-class generator as out-of-scope future work, rather than implying the conclusion is model-independent.
- **Why it lives here, not in Paper 1:** this is the direct answer to Jordan's "pivot to retrieval / no generation" critique — generation is a first-class contribution of the *system* paper, and it reaches the ~65% of items that are ungroundable for t_suf.

---

## Component C — Cross-retriever TIS × quality (new; Jordan's "very good metric")

For each retriever (BM25, dense, + a reranker), plot TIS (φ_suf, streamable fraction, `H`) against generated-answer quality on the same questions with the same generator.

- **Thesis to establish:** *a retriever with lower TIS at equal answer quality is strictly preferable for streaming* — same answers, more hideable latency. This operationalizes Jordan's idea and gives TIS **external (predictive) validity** beyond the intrinsic measure.
- **The thesis is in tension with our own two-retriever data — pre-register that.** Paper 1 already shows dense has *higher* quality (363 vs. 292 retrieved-gold) **and** *higher/worse* TIS (φ_suf 0.222 > 0.143) than BM25. So in the data we have, quality and TIS move together the "wrong" way, and a naive "pick the low-TIS retriever" rule would pick the *lower-quality* one. The reranker is the decisive third point: it tests whether one can buy dense-level quality at BM25-level TIS (the "equal quality, lower TIS" cell the thesis needs) or whether the trade-off is fundamental. We pre-register both outcomes — a clean trade-off curve, or a negative result that bounds when TIS is a valid retriever-selection signal (see kill-criteria).
- **Population confound (must control).** φ_suf is defined relative to *each retriever's own* retrieved-gold set, and those sets differ in size and difficulty (292 vs. 363). Comparing φ_suf across retrievers therefore compares different populations unless restricted to the **intersection** of retrieved-gold queries, scored with a common generator. That intersection is small — only **264 questions** are retrieved-gold under *both* BM25 and dense (of a 391-query union) — so it materially shrinks n; report it and run the cross-retriever comparison on it. State this protocol before any number is reported.

---

## Component D — Expanded RQ3 + H_corrected

**Motivation (lead with the failure we already have).** The dense replay in Paper 1 produces **net-negative perceived savings on 20% of queries (12/60)**, vs. just **1/60 (≈1.7%)** for the BM25 replay that Paper 1 actually reports — `H` is an aggregate floor the mean clears while a real per-query tail goes negative, and that tail is 12× heavier under dense retrieval than the paper's headline rate admits. The BM25 replay's single mis-fire was too small to characterize *that* failure mode. Component D's job is to characterize and shrink the negative-saving tail (which retrievers, which query types), not merely to add sample size.

- Expand the replay to **≥ 200** questions (retrieved-gold **and** t_sc-fallback populations); bootstrap CIs on mis-fire rate, net-negative rate, and mean saving; report the full distribution of per-query saving (not just the mean), so the negative tail is visible.
- **H_corrected = H + query_generation_constant** (≈ 590 ms, from the harness `Config`). Paper 1 showed measured saving exceeds `H` because the pipeline also hides query-generation time. **Framing matters and is a known trap:** `CLAUDE.md` warns that folding `query_gen_ms` into the hideable budget makes RQ3 near-circular. We therefore use H_corrected strictly as a **post-hoc explanatory** model — fit/justified from the `Config` constants, then validated against *held-out* measured savings — never as the bound we then declare the system to "meet." Test whether H_corrected predicts per-query saving better than `H` on held-out replays, and show the classifier's fire point tracks t_suf closely enough to realize most of H_corrected.

---

## Component E — Second benchmark (generalization)

Add Natural Questions or TriviaQA. Goal is generalizability of *the trigger's latency/quality wins*, not a new measurement: the φ_suf distribution, the question-type ordering, and the trigger's wins should replicate. If they don't, that is itself a result and the EMNLP-main "does it hold beyond CRAG?" question is answered with data.

**These benchmarks are not drop-in — say what ports and what is re-derived.** Neither NQ nor TriviaQA carries CRAG's question-type taxonomy or its web-page retrieval contents, and both skew to short, single-entity answers — which tends to push groundability *up* and put φ_suf in a different regime. State up front which pipeline pieces are reused (prefix streaming, t_suf grounding, the trigger) and which must be re-derived or dropped (the question-type breakdown may be unavailable; the retrieval corpus differs). "It should replicate" without this caveat reads as naive about the benchmarks' structure.

---

## Resource & latency evaluation (Elroy's "latency + resource consumption")

Report both axes, not just latency — and recall they trade off (Component A, competing-objectives):

- **Perceived-latency saved** (ms) — the user-facing benefit; report the full per-query distribution including the negative tail (Component D), not just the mean.
- **Retrieval-call count / wasted-retrieval rate** — fixed-interval fires every δ and issues many speculative retrievals; the classifier fires ~once near t_suf. Quantify the **compute / cost** reduction as a distinct selling point, and plot it *jointly* with latency saving (the Pareto frontier), since pushing one can cost the other.
- **Trigger cost itself** — confirm the classifier is negligible (CPU-only, sub-ms), so net compute clearly favors it.

---

## Stretch — H5 cross-linguistic (only if A–E are solid)

In SOV (verb-final) languages, predicate-hinged queries should stabilize *later* under BM25 (verb arrives last) but *earlier* under dense retrieval; case morphology partially restores early BM25 stabilization. Needs a **natively authored** Japanese/Korean QA set (JAQKET / KorQuAD — do **not** machine-translate CRAG), a Japanese/Korean BM25 (MeCab), a multilingual dense retriever, and ideally a native-speaker collaborator. High upside (psycholinguistics + multilingual + systems) but out of the critical path.

---

## Contribution statement (Paper 2)

Building on *tool-intent stabilization* — a retrospective measure of when a streaming query's retrieval converges on the answer-bearing document [Paper 1] — we present a live trigger that decides *when* to fire speculative retrieval from prefix-observable signals alone (running top-1 stability, named-entity position, word count). We show that (i) a lightweight, CPU-only classifier trained on φ_suf labels predicts P(t ≥ t_suf) and fires earlier on early-stabilizing queries than a fixed-interval baseline, tracing a latency–compute frontier rather than claiming both maxima at once; (ii) **end-to-end, triggered retrieval preserves generated-answer quality (for a fixed generator) while cutting perceived latency and retrieval calls**, with a Reflector bounding the cost of premature fires; and (iii) across retrievers, *lower TIS at equal answer quality* identifies the better streaming retriever where such a retriever exists — a relationship we test rather than assume, given that our own data show quality and TIS can move together. Our trigger is model-agnostic, interpretable, training-free at inference, and CPU-reproducible — a principled, lightweight alternative to the LLM fine-tuning of Stream RAG, whose target (downstream QA accuracy) entangles retriever and model quality and which we position against on design rather than on a head-to-head accuracy claim. Results hold on CRAG and [NQ / TriviaQA] under both sparse and dense retrieval.

---

## Timeline (dense removed — it's in Paper 1; generation added)

| Component | Work | Duration |
| :---- | :---- | :---- |
| A | Classifier trigger: split-1 label generation, design, features, train/eval, label audit | 4–5 weeks |
| B | End-to-end generation + answer-quality eval (incl. building the judged-accuracy harness) | 4–5 weeks |
| C | Cross-retriever TIS × quality (overlaps B) | 2 weeks |
| D | Expanded RQ3 (≥200) + H_corrected | 2–3 weeks |
| E | Second benchmark (parallel with D) | 2–3 weeks |
| — | Write-up | 2–3 weeks |
| **Total (A–E)** |  | **~14–16 weeks** |
| H5 | Cross-linguistic stretch | +4–6 weeks if pursued |

The estimate is deliberately wider than a first pass would suggest: Component B's CRAG-style judged accuracy needs an LLM-judge harness that does not exist yet, and Component A now includes a label-quality audit and split-1 label generation. Treat 14–16 weeks as the realistic A–E band.

---

## De-risking / kill-criteria

- **If triggered firing degrades answer quality** beyond a small margin vs. the full-query baseline → the headline claim fails; fall back to "trigger + Reflector recovers quality" and report the recovery cost honestly.
- **If cross-retriever TIS does not track quality** (the live risk, given dense already shows higher quality *with* higher TIS) → reframe C as a caveat/negative result that bounds *when* TIS is a valid retriever-selection signal — still informative.
- **If the trigger cannot beat fixed-interval on the latency–compute frontier** (only ties) → reframe around Component C / the metric's external validity as the primary contribution (a Findings-grade result), per the venue fallback.
- **Generator choice:** fix one small open instruct model up front; sweeping models reintroduces the confound Paper 2 exists to avoid.

---

## What to write first

A design doc before any code (it pre-empts the first reviewer questions):

1. Enumerate features **observable at inference** vs. retrospective-only — the core distinction.
2. Confirm the label: per-word binary `t ≥ t_suf`; specify that split-1 labels must be generated by re-running the Paper-1 pipeline (they are not in the existing split-0 CSVs).
3. Resolve the **population problem**: define trigger behavior on the ~65% ungroundable queries (t_sc safety target), and commit to reporting the two populations separately.
4. Operationalize **correct fire vs. mis-fire**: a fire at *t* is correct if `t ≥ t_suf`; a mis-fire if `t < t_suf` (Reflector recovers) or so late that the `H` saving is negligible.
5. Fix the **loss weighting** as a point on the latency↔compute frontier, and justify the chosen operating point.
6. Plan the **ablation**: hold out one feature at a time, measure Spearman ρ drop — retrieval-stability vs. entity-position is the hypothesis.
7. Define the **label-quality audit** (sample size, adjudication protocol, reported precision).
8. Define the **generation eval protocol** (generator, scoring, populations) and the **cross-retriever comparison protocol** (intersection population, common generator) before coding — these are the two new, confound-prone pieces.

---

## Venue target

- **A–E:** EMNLP main or ACL main (long paper) — a system contribution with downstream evaluation.
- **+ H5 confirmed:** strengthens an ACL-main case (psycholinguistics + multilingual + systems).
- **Fallback if generation quality is noisy or the trigger only ties fixed-interval:** reframe around the cross-retriever TIS × quality analysis (Component C) as the primary contribution — still a Findings-grade result.
