**Paper Review**

**When Does Streaming Tool Use Help? Characterizing Tool-Intent Stabilization in Streaming RAG**

**Author:** Elroy Galbraith, PhD (SMG Labs)

**Source:** Preprint — github.com/elroy-galbraith/stablize\_CRAG

**Reviewed:** June 18, 2026

**Lens:** Author-mode pre-submission, peer-review rigor.

**Bottom line.** A genuinely useful reframing with exemplary honesty, but the cleanest results live on a 21.3% slice whose selection bias is unbounded, and the system-validation RQ is inconclusive at the one operating point replayed. Both are fixable with your own planned follow-ups. Verdict: Major Revision (close to Accept at a workshop). Top three: (1) re-run RQ3 at L=1000 ms where H varies; (2) bound the grounding selection bias; (3) align the abstract with the body.

# **Summary**

The paper introduces tool-intent stabilization — a query-intrinsic, model-agnostic measure of when, within an utterance, a speculative retrieval converges on the answer-bearing document — and uses it to derive a bound H on how much tool latency streaming RAG can hide behind a user’s remaining input. On CRAG (1371 validation questions, BM25, training-free, CPU-only), it reports that retrieval-sufficiency stabilizes early on a groundable subset, that question type predicts stabilization, and that a calibrated working-pipeline replay realizes savings meeting or exceeding H. The stated contribution is measurement and analysis rather than a new system, and the work is unusually candid about its own limits.

# **Critical Issues**

| \# | Section | Issue | Severity |
| :---- | :---- | :---- | :---- |
| 1 | Results / Abstract | External validity. The marquee sufficiency results (φ\_suf=0.26, early stabilization, 95.2% streamable) are conditional on the 21.3% “retrieved-gold” slice — the intersection of verbatim-groundable ∩ BM25-surfaceable. The paper itself says this selects for an early lexical anchor and biases stabilization early, yet the magnitude of that bias is never bounded. The headline early-stabilization finding may be substantially a selection artifact. | **Major** |
| 2 | Results §5.4 (RQ3) | System validation is inconclusive by construction. At L=600 ms the bound saturates (59/60 queries share H=600), so there is no variance to test per-query prediction (Spearman ρ=−0.06). The “1.9× exceeds bound” is arithmetically the omitted query-generation constant: mean H=590 \+ query-gen≈590 ≈ measured 1122 ms. So RQ3 recovers H \+ a constant rather than independently validating H. One of four headline RQs is effectively untested. | **Major** |
| 3 | Results §5.3 / Abstract | The headline 73.9% “admit latency hiding” blends a 21.3% evidence-verified minority with a 78.7% majority where t\* falls back to t\_sc (top-1 merely stopped moving, not gold-in-top-k). For that majority, “streamable” means latency hidden behind a possibly-wrong query — and since H floors at 0 and (your words) “cannot express the negative saving a real mis-fire incurs,” the statistic counts upside but is blind to downside, precisely where mis-fires are most likely. The body decomposes this well; the framing still overcounts benefit. | **Major** |
| 4 | Abstract vs body | Overclaim. The abstract calls question type “an ordered predictor of stabilization,” but §5.2 shows bootstrap CIs overlap for all adjacent types and only the extremes separate — and the latest type, “set,” is n=2. Use the body’s honest phrasing (“significant type-level effect with a robust early/late split, not a precise per-type order”) in the abstract. | **Minor** |
| 5 | §3 (Hypotheses) | “Pre-registered” is asserted with no registration artifact or timestamp. For a solo-author study on a public benchmark this is a credibility liability — and of the four, H1 was falsified (right-skewed, not bimodal), H3 was contradicted in direction, H4 deferred; only H2 was cleanly confirmed. Either link the registration or soften to “pre-specified.” | **Minor** |
| 6 | §5.4 (Volatility V) | V is introduced as the early-commit / mis-fire signal but then shown not to flag the one observed failure (V=0 on the negative-saving query), and its “actionable role… remains untested.” A construct that earns a symbol but no demonstrated predictive value should be validated on a larger late-stabilizing sample or demoted. | **Minor** |

*Severity: **Major** affects core claims/validity · **Minor** weakens but doesn’t invalidate.*

# **Suggestions**

| \# | Section | Suggestion | Category |
| :---- | :---- | :---- | :---- |
| 1 | §5.4 (RQ3) | Re-run the replay at L=1000 ms (and/or sweep L) where H varies across queries, then report Spearman between residual (n−t\*)/δ and realized saving. Converts the acknowledged-inconclusive RQ into a genuine per-query validation — the single highest-impact change. | Methodology |
| 2 | §4 / §5 | Add a relaxed-grounding arm (fuzzy / embedding match for d\*) to recover part of the \~64% currently excluded, and report how φ\_suf shifts. This bounds the selection bias that caps the paper’s reach. | Robustness |
| 3 | §5.3 / Abstract | Lead with the scoped, verified numbers (95.2% on the retrieved-gold subset; the early-sufficiency / late-self-consistency gap) and label 73.9% explicitly as top-1-settling-dominated. Pair the streamable fraction with a downside statistic (estimated mis-fire / negative-saving rate). | Framing |
| 4 | §2 (Related work) | Connect to simultaneous translation / incremental NLP — wait-k policies, prefix-to-commit decisions, incremental dialogue processing — where “stabilization of a prediction over a growing prefix” is a long-studied problem. Strengthens novelty positioning and the thin 6-reference related work. | Related Work |
| 5 | §4 (Reproducibility) | Pin BM25 params (k1, b, implementation), chunk overlap, the CRAG snapshot date (web pages are time-sensitive), and seeds for the 60-question subset and the 10k bootstrap. “Every metric recomputable offline” requires the retrieval config to be stated. | Reproducibility |
| 6 | §3 / §5.2 | Add an effect size for the Kruskal–Wallis (ε²) and a corrected post-hoc (Dunn’s) instead of eyeballing overlapping CIs, to support the early/late split formally. | Statistics |
| 7 | Intro / Abstract | Reconcile the “decisive term first vs last” bimodal intuition with the right-skewed finding (H1 falsified); the narrative still leans bimodal. | Clarity |

# **What the Paper Does Well**

* **A useful reframing.** Separating query-intrinsic headroom from system quality is the real contribution: streaming RAG’s benefit is upper-bounded by where the decisive information sits in the utterance (“who makes the console called the PlayStation”). Crisp and actionable.

* **Exemplary intellectual honesty.** The paper pre-empts most of its own critiques — the 21.3% slice, string-grounding false positives, the saturated bound, the negative-saving case, overlapping CIs. Decomposing 73.9% into 95.2% / 68.1% is exactly the right move.

* **A clean empirical result.** Early sufficiency (φ\_suf=0.26) vs late self-consistency (φ\_sc=0.67), with a plausible entity-position mechanism that sensibly refines (rather than rescues) the original hypothesis.

* **Counterintuitive and internally consistent.** The H2 result (slower cadence raises the hideable fraction for large L) is well-explained, and the arithmetic checks out — mean H=590 follows exactly from 59×600 \+ 1×0 over 60 queries.

* **Low-friction reproducibility.** Training-free, CPU-only, with the derived d\* labels released — and refreshingly modest about being measurement, not a system.

# **Questions for the Authors**

1. Is there an actual pre-registration (link / timestamp), or should “pre-registered” become “pre-specified”?

2. How much does φ\_suf move if grounding is relaxed (embedding / fuzzy) to recover the excluded \~64% — can you bound how much of the early-stabilization result is selection vs signal?

3. On the t\_sc-fallback majority inside the 73.9%, what is the estimated mis-fire rate, and what would a Reflector rejection (re-fire) cost do to perceived latency there?

4. At an unsaturated operating point (L=1000 ms), does H actually rank queries by realized saving — the per-query claim RQ3 could not test?

# **Verdict**

**Major Revision**  —  close to Accept for a workshop; Major for a main track.

**Rationale.** The conceptual contribution is real and the transparency is exemplary, but the cleanest empirical claims are confined to a 21.3% slice whose selection bias is unbounded, and the system-validation RQ is inconclusive at the operating point chosen. Both are fixable with the author’s own planned follow-ups (higher-L replay, relaxed grounding) and tighter abstract/body alignment. The core idea — that streaming’s benefit is a measurable property of the query, not the system — is worth publishing once the empirical envelope matches the framing.