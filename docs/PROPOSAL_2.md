# Research Plan: From Measurement to Main Venue
## Extending *Tool-Intent Stabilization* with a Classifier Trigger

**Project:** stablize_CRAG  
**Current state:** Short-paper / workshop submission measuring φ, H bound, RQ1–RQ4 on CRAG/BM25  
**Target:** Main-venue long paper (EMNLP or ACL)  
**Core new contribution:** A lightweight classifier that learns to recognize, from prefix-observable signals alone, when retrieval sufficiency has probabilistically been reached — bridging the gap between the retrospective φ_suf measure and a live streaming trigger

---

## What the Paper Currently Has vs. What It Needs

| | Current paper | Main-venue paper |
|---|---|---|
| Retriever | BM25 only | BM25 + dense (closes confound) |
| Benchmark | CRAG (English) | CRAG + one more (generalizability) |
| System contribution | None (analysis only) | Classifier trigger that outperforms fixed-interval |
| RQ3 sample | 60 questions | 200+ questions |
| Language | English only | English core; SOV as stretch |

The classifier trigger is the pivot. Without it this is a measurement paper. With it, you have the loop: *measure stabilization → train classifier on φ_suf → deploy trigger → show it beats fixed-interval*. That's a system contribution reviewers can evaluate.

**Positioning vs Stream RAG:** Arora et al. use a post-training pipeline that fine-tunes the LLM itself to decide when to fire — trained on downstream QA accuracy signals. That confounds retriever quality, model quality, and query difficulty, and requires LLM fine-tuning. Your classifier is trained on φ_suf labels — *retrieval sufficiency at the prefix level* — a query-intrinsic signal computable without the LLM. Stream RAG never defines this target at all. Your trigger is model-agnostic, interpretable, CPU-only at inference, and grounded in the theoretical framework. Position it as the principled, lightweight foundation that explains *why* a trigger should fire, rather than as a competitor to Stream RAG's heavier learned system.

---

## Phase 1: Dense Retriever (3–4 weeks)
**Why first:** BM25 fires early on keyword anchors. If φ_suf is low only because BM25 matches the topic word in the first three tokens, the early-stabilization finding is a retriever artifact, not a query property. You need to show it holds under dense retrieval before building a classifier on top of it.

**What to do:**
- Add `all-MiniLM-L6-v2` (or `multilingual-e5-small` if you want to hedge toward H5) as a second retriever in `experiments/stabilization.py`
- Re-run the full stabilization sweep; compute φ_sc and φ_suf under dense retrieval
- Add a comparison table: BM25 vs. dense on the four RQ metrics

**Key question to answer:** Does the BM25 vs. dense ordering of question types agree? If aggregation/comparison still stabilize earliest under dense retrieval, the entity-position account survives and the classifier has a solid foundation. If the ordering flips, the account needs revision before you build on it.

**Deliverable:** Updated Table 1 / Figure 2 with a BM25/dense column. This also lets you retire the "BM25 confound" limitation from the current paper.

---

## Phase 2: Classifier Trigger (4–5 weeks)
**The core idea (yours):** The paper measures t_suf *retrospectively*. The classifier predicts, at each word position t in a live stream, whether retrieval sufficiency has probably been reached — so the Trigger can fire as soon as the classifier is confident, rather than on a fixed interval.

### The Observability Problem (and Why It's Not Fatal)

φ_suf = t_suf / n. In a live stream, neither n (total query length) nor d* (the gold document) is known. You cannot directly observe whether t ≥ t_suf at inference time. This is the fundamental challenge.

But it is the *standard* setup for learned commit classifiers. The label is retrospective; the prediction is prospective. The classifier learns the mapping:

> **observable proxy signals at time t → P(t ≥ t_suf)**

This is identical in structure to simultaneous MT commit classifiers (which learn to predict "enough source seen" without knowing the sentence-final verb) and incremental SLU (which predicts intent without knowing the full utterance). The label is retrospective; only the features need to be live-observable.

Your RQ4 result provides the inductive bias: entity position is already the dominant predictor of t_suf. The classifier is essentially learning to recognize when the named entity anchor has appeared — which is directly observable in the live stream. The theoretical account from the paper *becomes* the classifier's feature design.

Critically, the classifier does not need to be perfect because **the Reflector is still in the loop**. If the classifier fires at t and retrieval misses d*, the Reflector catches it and re-fires. The cost of a false positive is one wasted retrieval call, not a wrong answer. This asymmetry should inform the loss function: **penalize late fires more than early fires**, since a missed latency saving is unrecoverable but a premature fire is correctable.

### Framing

At word position t, observe prefix features and output:

> P(t ≥ t_suf | features at t)

Fire when P exceeds a threshold. The classifier is trained offline on retrospective φ_suf labels; at inference time it only sees the observable features below.

### Features

Two categories — both observable in the live stream:

**Retrieval-side (requires running BM25 on each prefix):**

| Feature | Rationale |
|---|---|
| `top1_stable_streak` | Consecutive prefixes with same BM25 top-1 — t_sc proxy, observable without knowing d* |
| `top1_changed_since_last` | Did d_t change vs d_{t-1}? Volatility → not yet stable |

**Query-side (from the token stream only, no retrieval needed):**

| Feature | Rationale |
|---|---|
| `t` (absolute word count) | Unconditional: more words → higher P(t ≥ t_suf). Use absolute t, not t/n — n is unknown at inference |
| `named_entity_detected` | Named entity in q_{1:t} = topic anchor present; directly observable |
| `entity_word_position` | Word position of first NE — early anchor → early stabilization (your RQ4 result) |
| `question_word_type` | who/what/when/where/which from first token → entity type prior |

Keep it to ≤6 features. The model should be logistic regression or GBDT — interpretable, CPU-only, trainable in seconds. The feature importance ablation is a main result: it tests whether entity-position or retrieval-stability is the dominant signal, connecting back to the psycholinguistics framing in §2.

Note: **do not use t/n as a feature** — n is unknown at inference time. Use absolute word count t. The classifier learns the absolute-t distribution from training data implicitly.

### Training and evaluation

- **Labels:** per-word binary: 1 if t ≥ t_suf, 0 otherwise (derived directly from existing per-question CSV)
- **Train split:** CRAG split 1 (~1,335 questions, the non-validation portion)
- **Test:** expanded RQ3 replay set (200+ questions — see Phase 3)
- **Loss:** weighted binary cross-entropy, upweighting the positive class (t ≥ t_suf) to penalize late fires more than early fires

### Evaluation metrics

Compare classifier trigger vs. fixed-interval trigger (current baseline) and vs. Stream RAG's LLM-based trigger (if results are available):

| Metric | What it captures |
|---|---|
| Mean perceived-latency saving (ms) | Upside benefit |
| Net-negative rate (%) | Mis-fire downside (Reflector-correctable) |
| Median saving at correct fires | Quality of the hits |
| Spearman ρ (fire word t vs. actual t_suf) | Per-query ranking quality |
| Feature importance (ablation) | Which signal drives the classifier |

The key result: *the classifier fires earlier on early-stabilizing queries and later on late-stabilizing ones*, improving mean saving while keeping mis-fire rate comparable to fixed-interval.

### On the negative outcome

The current paper has one mis-fire out of 60 questions — too few to characterize the failure mode. With 200+ in the expanded replay you'll have enough to measure the net-negative rate reliably and test whether the Reflector recovery cost is bounded.

---

## Phase 3: Expand RQ3 and Close the Loop (2–3 weeks)

The current RQ3 has n=60 from the retrieved-gold population. For the main paper you need:
- Expand to 200+ questions (both retrieved-gold and t_sc-fallback populations)
- Bootstrap CIs on mis-fire rate and mean saving
- Ablation: which feature(s) matter most? (entity-position vs. top1_stable_streak are the hypotheses)
- Show: classifier trigger outperforms fixed-interval on saving, and matches or beats it on mis-fire rate

This is also where the H bound gets its corrected version: H_corrected = H + query_generation_constant (≈590ms). Show that H_corrected predicts per-query saving better than H, and that the classifier trigger's fire point tracks t_suf closely enough to realize most of H_corrected.

---

## Phase 4: Second Benchmark (2 weeks, run in parallel with Phase 3)

Add Natural Questions or TriviaQA. The goal is generalizability, not a new finding — the question-type ordering and φ_suf distribution should replicate. If they don't, that's also a result.

This is table insurance. Reviewers at EMNLP main will ask "does this hold beyond CRAG?" You want to answer yes with data, not hope.

---

## Phase 5: Cross-Linguistic H5 (stretch, only if Phase 1–4 are solid)

**The prediction:** In SOV (verb-final) languages (Japanese, Korean), queries that hinge on the predicate stabilize later under BM25 (verb arrives last, BM25 misses it) but earlier under dense retrieval (semantic intent captured without waiting for the verb). Case morphology partially restores early stabilization even for BM25.

**Why it's high upside:** This is a falsifiable, cross-disciplinary claim that nobody has tested. If it holds, you have a paper that speaks to psycholinguistics, multilingual NLP, and systems audiences simultaneously.

**What you'd need:**
- A natively authored Japanese or Korean open-domain QA benchmark (JAQKET for Japanese, KorQuAD for Korean). Do *not* machine-translate CRAG — translation artifacts confound the word-order signal.
- Japanese/Korean BM25 (MeCab + BM25 for Japanese; Mecab-ko for Korean)
- Dense retrieval: `multilingual-e5-small` or `paraphrase-multilingual-MiniLM-L12-v2`
- A collaborator with native-language NLP expertise is strongly recommended

**If H5 is out of scope for now:** The entity-position account (§5.2 finding) is the English-only version of this claim. Quantify it directly: measure word-position of the first named entity vs. t_suf across question types. If entity-word position predicts t_suf better than question type (ε² > 0.04), you've closed RQ4 and strengthened the classifier's theoretical basis.

---

## Revised Contribution Statement (target paper)

> We introduce *tool-intent stabilization*, a query-intrinsic, model-agnostic measure of when a streaming speculative retrieval can fire without sacrificing accuracy. Critically, φ_suf — the sufficient proportion of the query at which the answer-bearing document becomes retrievable — is a retrospective measure: it cannot be directly observed in a live stream where neither the total query length n nor the gold document d* is known. We bridge this gap with a lightweight classifier trigger that learns, from prefix-observable signals alone (named-entity position, running retrieval top-1 stability, word count), to predict P(t ≥ t_suf) prospectively. On CRAG and [benchmark 2], with both BM25 and dense retrieval, we (i) characterize the φ_suf distribution and show it is governed by entity position in the utterance, (ii) derive and validate a conservative latency-hiding bound H, and (iii) show the classifier trigger outperforms a fixed-interval baseline on mean latency saving while keeping mis-fire rates low, with the Reflector bounding the cost of false positives. Unlike the LLM fine-tuning approach of Stream RAG — which trains on downstream QA accuracy and confounds retriever and model quality — our classifier is trained on φ_suf, a retrieval-sufficiency signal that is model-agnostic, interpretable, and grounded in the paper's theoretical framework. The full pipeline is training-free at inference time and CPU-reproducible.

---

## Timeline Summary

| Phase | Work | Duration |
|---|---|---|
| 1 | Dense retriever: replicate RQ1–RQ4, close BM25 confound | 3–4 weeks |
| 2 | Classifier trigger: design, features, train/eval on expanded harness | 4–5 weeks |
| 3 | Expand RQ3 to 200+, ablations, H_corrected | 2–3 weeks |
| 4 | Second English benchmark (run in parallel with 3) | 2 weeks |
| 5 | Cross-linguistic H5 (stretch) | 4–6 weeks if pursued |
| — | Write-up | 2–3 weeks |
| **Total (Phases 1–4)** | | **~13–15 weeks** |

---

## What to Write First

Start with the classifier design doc before touching code:
- Enumerate which features are observable at inference time vs. retrospective-only (this is the core distinction and will be the first reviewer question)
- Confirm the training label: per-word binary (t ≥ t_suf), with t_suf derived from the existing CSV
- Define "correct fire" vs. "mis-fire" operationally: a fire at word t is correct if t ≥ t_suf; it is a mis-fire if t < t_suf (Reflector recovers it) or if it fires so late that H saving is negligible
- Define the loss weighting explicitly before training — the asymmetry (late fires worse than early) is a design choice that needs justification
- Plan the ablation: hold out one feature at a time, measure Spearman ρ drop. Entity-position vs. top1_stable_streak is the hypothesis to test

Getting this on paper first keeps the classifier theoretically grounded. The feature importance ablation is a main result — it tests whether the psycholinguistic account (entity position → sufficiency) or the retrieval-stability signal is what the classifier actually learns to use.

---

## Venue Target

- **With Phases 1–4:** EMNLP main or ACL Findings (long paper)
- **With Phase 5 (H5 confirmed):** ACL main
- **Fallback if dense retrieval diverges from BM25:** reframe as a retriever-comparison study; still publishable at EMNLP Findings with the classifier
