# Paper Review: When Does Streaming Tool Use Help?

**Author:** Elroy Galbraith, SMG Labs
**Source:** Preprint (intended for arXiv)
**Reviewed:** June 18, 2026
**Mode:** Author / pre-submission coaching

---

## Bottom line

The core idea is genuinely good and the framing is honest. But there's a dimensional error in the central equation, the two headline numbers overclaim their scope, and the one system-validation doesn't actually test the thing it claims to. None are fatal — but do not post to arXiv until at least the three cheap-but-serious issues are fixed (#1, #3, #6) plus the abstract wording (#5).

---

## Summary

The paper introduces *tool-intent stabilization* — the point in a streaming utterance at which a speculative retrieval query converges on the answer-bearing document — and argues that the benefit of Streaming RAG is *query-intrinsic*: a property of where the decisive token sits, not of the model or retriever. On CRAG it measures two stabilization notions (self-consistency vs. sufficiency), derives a hidden-latency bound *H*, reports the streamable fraction of the workload, and shows question type predicts stabilization. The contribution is explicitly measurement, not a new system.

---

## Critical Issues

| # | Section | Issue | Severity |
|---|---------|-------|----------|
| 1 | Eq. (1) / §4 | **Dimensional inconsistency in the core definition.** Eq. 1 defines δ as "per-token arrival interval" (sec/token), giving (n−t⋆)·δ = seconds. But everywhere else δ ∈ {2,3,4} **w/s** (a rate). With δ in w/s, (n−t⋆)·δ is dimensionally wrong and Table 2 can only have been computed as (n−t⋆)/δ. The H2 narrative ("slower cadence raises streamable fraction") confirms the division form. As printed, the central equation is wrong. | 🔴 Major |
| 2 | Abstract / §5.1 | **"Sufficiency stabilizes early for most answerable questions" overstates scope.** φ_suf is computed on the *retrieved-gold* population — 21.3% of the benchmark (n=292), the subset that is both verbatim-groundable *and* BM25-friendly. That's a favorable, non-random slice, not "most answerable questions." The selection effect plausibly biases stabilization *earlier*. | 🔴 Major |
| 3 | §5.3 | **The 73.9% headline is a blended metric.** You use t_suf for 21% of queries and fall back to t_sc for the other 79%. So the central deployment number mostly reflects self-consistency — the notion you explicitly deprioritize — and "streamable" means two different things (evidence-in-hand vs. top-1-stopped-moving) across the population. | 🔴 Major |
| 4 | §5.2 / Table 1 | **"Strong, ordered predictor" is asserted without any uncertainty quantification.** For a paper whose contribution *is* measurement, there are no CIs or tests anywhere. The ordering interleaves cells with n=2 (set) and n=5 (post-processing), reported to three decimals. You need a test across types (Kruskal–Wallis / bootstrap CIs) before claiming a strong ordering. | 🔴 Major |
| 5 | §5.4 / Fig. 4 | **RQ3 doesn't validate per-query prediction.** At L=600 the *H* bound saturates, so nearly all points share one x-value — there's no horizontal spread to demonstrate *H* predicts cross-query variation. You show measured > bound (a constant-ish offset + noise would too), not that *H* *predicts* savings. And measured exceeding *H* contradicts the abstract's "**upper** bound on achievable latency savings" — *H* is a bound on the *input-hiding component*, not on total savings. | 🔴 Major |
| 6 | §5 | **H1–H3 are referenced but never stated.** "confirming H2," "refines H3," "as H1 anticipated" — the hypotheses are never defined. A hypotheses block is missing (cut for space?). | 🟡 Minor |
| 7 | §4 | **String-grounding false positives unaddressed.** Substring matching means single-token gold answers (years, numbers, yes/no) can match a passage *coincidentally*, registering sufficiency before the real evidence arrives — biasing t_suf early. You discuss the groundable-rate limitation but not match precision. | 🟡 Minor |
| 8 | §3 / §5.4 | **Volatility V is defined, then orphaned.** It's introduced as "early-commit risk" but never linked to outcomes — notably the negative-saving mis-fire queries in Fig. 4, which is exactly where V should earn its place. H's max(0,·) also means the model structurally can't predict the downside you observe. | 🟡 Minor |
| 9 | §4 | Tokens vs. words used interchangeably (q_{1:n} in tokens, δ in words/sec). Subsumed by #1 but worth a consistent-units pass. | 🔵 Nitpick |

---

## Suggestions

| # | Section | Suggestion | Category |
|---|---------|------------|----------|
| 1 | §4 | Report the retrieved-gold subset's properties vs. the excluded 79% (length, type mix) so readers can judge the selection effect's direction. | Validity |
| 2 | §5.3 | Report the streamable fraction **separately** for the t_suf and t_sc populations, then the blend. Let the honest sufficiency-only number stand on its own. | Clarity |
| 3 | §5.4 | Re-plot Fig. 4 at L=1000 (or color by L) so *H* varies across queries and the correlation is testable. Report a rank correlation. | Evidence |
| 4 | — | Release the derived d⋆ labels + harness code. It's training-free and CPU-only — reproducibility is nearly free and would make this far more citable. | Reproducibility |
| 5 | §4 | Specify the Arora et al. per-stage latencies used to calibrate RQ3; right now RQ3 can't be reproduced. | Reproducibility |

---

## What the Paper Does Well

- **The central reframing is the real contribution and it's clean:** "achievable latency hiding is a property of where the decisive token sits" is a genuinely useful, model-agnostic lens, and you're honest that it's measurement not a system.
- **Separating self-consistency from sufficiency** is the right conceptual move, and the 0.67-vs-0.26 gap is a crisp, memorable result.
- **The question-type reinterpretation** (stabilization tracks *entity position*, not reasoning complexity — so aggregation/comparison stabilize early) is the most interesting finding in the paper, *if* it survives a significance test.
- Threats-to-validity is candid and well-organized.

---

## Questions for the Authors

1. Within the retrieved-gold subset, does φ_suf correlate with where the answer *string* first appears in the utterance? If yes, you've got a much stronger, mechanism-level version of the entity-position claim — and a check on grounding false positives.
2. What does the streamable fraction look like for sufficiency *only* (drop the t_sc fallback)? That's the number that matches your own framing.
3. Are the negative-saving queries in Fig. 4 high-V? If so, V becomes predictive of when *not* to speculate — a much more actionable result than its current cameo.
4. How sensitive is the type ordering to dropping the n<10 classes? If aggregation/comparison/simple/simple_w_condition alone carry it, say so and lead with them.

---

## Verdict

**Major Revision.**

**Rationale:** Strong, honest core idea undercut by a dimensional error in the central equation, two headline statistics that blend populations and overclaim scope, and a validation that doesn't test per-query prediction. All are addressable, several cheaply.

---

## On arXiv specifically

No peer-review gate, so the bar is "credible, not embarrassing." **Not as-is** — posting with a dimensional error in Eq. 1 and an abstract that says "upper bound" when the data shows measured savings *exceed* it is the kind of thing that gets screenshotted. Fix #1, #3, #5, and #6 (a few hours' work) and it's postable. Fixes #2 and #4 are what make it genuinely solid and workshop-submittable.
