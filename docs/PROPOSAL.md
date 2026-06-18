**When Does Streaming Tool Use Help?**

**Characterizing Tool-Intent Stabilization in Streaming Retrieval-Augmented Generation**

*Research Proposal*

Elroy Galbraith, PhD  ·  SMG Labs  ·  Draft v1

# **Abstract**

Streaming Retrieval-Augmented Generation (Streaming RAG; Arora et al., 2025\) reduces user-perceived latency by issuing tool queries in parallel with ongoing user input, before the utterance is complete. Reported gains are aggregate, yet the mechanism's benefit is fundamentally *query-intrinsic*: speculation can only help when the correct tool query becomes determinable before the user stops speaking or typing. This proposal isolates and measures that property, which we call **tool-intent stabilization** — the point in the input stream at which a speculative query's retrieval converges to the answer-bearing result. We will measure its distribution across a realistic QA workload (CRAG), derive a model-agnostic upper bound on achievable latency savings as a function of tool latency and input cadence, validate that bound against a working streaming pipeline, and identify the query properties that predict early versus late stabilization. We situate the measure within established work on incremental language comprehension and simultaneous translation, and extend it cross-linguistically: in verb-final (SOV) languages the cue that drives stabilization is predicted to shift from word order to early case morphology. The study requires no model training and runs on commodity CPU hardware, using an existing reproduction harness.

# **1\. Background and Motivation**

End-to-end and tool-augmented dialogue systems increasingly rely on external retrieval to remain factual. Tool calls, however, add latency that disrupts conversational flow. Streaming RAG (Arora et al., 2025\) addresses this by predicting and issuing tool queries *in parallel with* the user's still-arriving input, then reflecting on whether retrieved results suffice. The authors report up to 200% relative QA-accuracy improvement and roughly 20% reduction in tool-use latency, and emphasize that the approach is modality-agnostic, applying equally to typed input.

The reported latency benefit is, however, an aggregate over a benchmark. It leaves a basic question unanswered: **for which queries can speculation help at all, and by how much?** The answer is not a property of the model, the retriever, or the speech stack — it is a property of where, within an utterance, the information that determines the tool query first appears. If the decisive term arrives last (“who makes the console called the *PlayStation*”), no early speculative query can retrieve the right evidence, and the latency win collapses to zero regardless of system quality. If the decisive term arrives early, almost the entire tool latency can be hidden behind the remaining input.

We propose to study this directly. By characterizing tool-intent stabilization as a measurable, query-intrinsic quantity, we obtain (i) a predictive upper bound on achievable latency savings *before* any system is built; (ii) a principled account of the failure mode; (iii) actionable guidance on whether a learned trigger is worth its cost; and (iv) results that transfer across text and speech because the quantity is modality-agnostic.

# **2\. Related Work and Positioning**

The phenomenon underlying tool-intent stabilization — that the meaning and referential intent of an utterance is fixed incrementally, often well before the utterance ends — is among the most robust findings in psycholinguistics. Altmann and Kamide (1999) showed that a verb such as “eat” triggers anticipatory attention to edible objects before they are named, so partial input already restricts what can follow. Kamide, Altmann and Haywood (2003) extended this cross-linguistically: in head-final Japanese, syntactic and semantic constraints from pre-verbal, case-marked arguments let comprehenders anticipate a forthcoming argument even before the verb (the head) is heard. This *pre-head anticipatory processing* (Kamide and Mitchell, 1999\) is the human analogue of firing a tool query from a prefix.

The engineering counterpart is simultaneous machine translation, which asks how much source input must be read before output can begin. The wait-k policy (Ma et al., 2019\) emits output a fixed number of tokens behind the source by learning to anticipate; word-order divergence is the central difficulty, and verb-final (SOV) source languages are the hard case because the predicate arrives last. Grissom et al. (2014) frame it as an agent that waits until it is confident enough to predict the sentence-final verb and only then commits — a direct precedent for our Trigger/Reflector design, including the failure mode in which premature commitment yields revisions that later input would have corrected (our volatility, *V*).

Closest to our measurement is incremental spoken language understanding. DeVault et al. (2011) produce a complete meaning frame from partial ASR and quantify understanding as a function of partial-input length; later incremental SLU work plots an “early detection” curve of intent accuracy over normalized utterance length and performs early intent detection before end-of-speech, within the incremental-unit framework (Schlangen and Skantze, 2011). These efforts establish that intent stabilizes mid-utterance and is measurable — but they target a *closed* intent / dialogue-act ontology. Our target differs: we measure stabilization of the *retrieval (tool) intent* — which external document the system would fetch — against an open-world knowledge benchmark, and we tie it to a tool-latency-hiding bound rather than to dialogue-act accuracy.

Positioned this way, our contribution is deliberately narrow. We do not claim to discover incremental intent determination. We identify that Stream RAG (Arora et al., 2025\) leaves the *when-to-fire* decision uncharacterized, import a prediction from incremental processing and simultaneous translation, and provide the first measurement of retrieval-intent stabilization as a query-intrinsic, cross-linguistically varying property that bounds achievable latency savings.

# **3\. Research Questions**

1. RQ1 (descriptive). How is tool-intent stabilization distributed across a realistic QA workload?

2. RQ2 (bound). What fraction of queries admit latency hiding as a function of tool latency L and input cadence δ?

3. RQ3 (validation). Does the perceived-latency saving measured from a working streaming pipeline match the bound predicted by stabilization?

4. RQ4 (explanatory). Which query properties — question type, length, position of the key entity — predict early versus late stabilization?

# **4\. Formalization**

Let a query Q be a token sequence q*₁₊ₙ* of length n, and let q*₁₊ₜ* denote the prefix of the first t tokens. Let R be a retriever returning a ranked document list; write d*ₜ* \= top-1(R(q*₁₊ₜ*)) for the top document retrieved from the prefix, d*ₙ* for the full-query top document, and d\* for the gold answer-bearing document (available in CRAG).

**Self-consistency stabilization point** *t\_sc* \= the smallest t such that for all s ≥ t, d*ₛ* \= d*ₙ*. (The prefix already retrieves what the full query will.)

**Sufficiency stabilization point** *t\_suf* \= the smallest t such that d\* ∈ top-k(R(q*₁₊ₜ*)). (The prefix already surfaces the gold evidence — the operationally meaningful notion, since the full query itself may retrieve imperfectly.)

**Stabilization fraction** *φ* \= t\*/n ∈ (0, 1\], reported for both t\_sc and t\_suf. Lower φ means earlier stabilization and more headroom to hide latency.

**Stabilization volatility** *V* \= the number of top-1 changes occurring after d*ₙ* is first reached. High V signals early-commit risk: a confident-looking early result that a later token would overturn.

**Hidden latency** *H(Q; L, δ)* \= min( L , max(0, (n − t\*) · δ) ), where L is tool latency and δ is the per-token arrival interval. H is the portion of tool latency that completes behind the user's remaining input — the achievable per-query perceived-latency saving.

**Streamable(Q; L, δ, θ)** holds iff H ≥ θ·L for a chosen coverage threshold θ (e.g., θ \= 0.8 hides at least 80% of tool latency). The streamable fraction of a workload is the central deployment-relevant statistic.

# **5\. Hypotheses**

* **H1.** The distribution of φ is bimodal: a large early-stabilizing mass (entity stated up front) plus a tail that stabilizes only near the end (decisive term last).

* **H2.** For large tool latency L, the binding constraint on H is residual input time (n − t\*)·δ, not L; thus slower input cadence δ paradoxically increases the hideable fraction.

* **H3.** CRAG question type predicts φ: simple lookups stabilize early; comparison, aggregation, multi-hop, and post-processing queries stabilize late.

* **H4.** Switching from sparse (BM25) to dense retrieval shifts absolute t\* but preserves the ordering of φ across question types.

* **H5 (cross-linguistic).** In verb-final (SOV) languages such as Japanese and Korean, queries whose tool intent hinges on the predicate stabilize later in linear word order than in SVO English; but early case morphology can restore earlier stabilization, so the driving cue shifts from word position to morphology. A lexical retriever (BM25), blind to case marking, will therefore stabilize later on SOV input than a morphology-aware or dense retriever.

# **6\. Data and Materials**

* **Queries:** CRAG, the Comprehensive RAG Benchmark (Yang et al., 2024), which provides QA pairs, gold answers, retrieval contents (web pages and a mock knowledge-graph API), and a question-type taxonomy. The companion AudioCRAG (TTS of CRAG) enables a later speech extension without changing the underlying intent.

* **Retrieval:** the CRAG-provided corpus, indexed with BM25 (sparse) and a sentence-embedding retriever such as all-MiniLM-L6-v2 (dense) for a retriever-robustness check. Both run on CPU.

* **System:** an existing async streaming-RAG harness implementing fixed-interval Trigger, parallel Threads, and a Reflector, with a swappable tool layer (in-process vs simulated MCP transport). Used for RQ3 validation.

* **Cross-linguistic (H5):** a natively authored verb-final QA set (e.g. a Japanese or Korean open-domain QA benchmark) for the extension. We deliberately avoid machine-translating CRAG into these languages, because translation artifacts would confound the very word-order signal under test (He et al., 2016).

# **7\. Method**

5. Stream each query as an ordered token/word sequence, a controlled proxy for input cadence δ.

6. For each prefix length t \= 1..n, run R(q₁₊ₜ) and record dₜ, the rank of dₙ, and the rank of d\*.

7. Compute t\_sc, t\_suf, φ, and V per query.

8. Sweep a grid of (L, δ); compute H and the streamable fraction (RQ2).

9. Run the streaming pipeline on the same queries; compare measured perceived-latency saving against the H-bound (RQ3).

10. Regress φ on query features and break results down by CRAG question type (RQ4).

# **8\. Experimental Design**

Factors varied:

| Factor | Levels |
| :---- | :---- |
| Retriever | BM25 (sparse); all-MiniLM-L6-v2 (dense) |
| Reference document | full-query top-1 (t\_sc); gold doc (t\_suf) |
| Top-k for sufficiency | 1, 3, 5 |
| Tool latency L | 100, 300, 600, 1000 ms |
| Input cadence δ | 2, 3, 4 words/sec |
| Coverage threshold θ | 0.5, 0.8, 1.0 |
| Query language | English (CRAG); native verb-final set (Japanese/Korean) — extension |

**Primary outcomes:** the distribution of φ (overall and per question type); the streamable fraction over the (L, δ, θ) grid; the absolute error between H-predicted and measured perceived savings; and volatility V as an early-commit risk indicator.

# **9\. Expected Contributions**

* A precise statement of a gap in Stream RAG (Arora et al., 2025): the decision of when to fire a speculative tool call is left uncharacterized, although it governs whether the method helps at all.

* A cross-field transfer: we import the incremental-processing and simultaneous-translation account of incremental intent determination into streaming tool use, yielding a falsifiable, query-intrinsic prediction rather than an aggregate benchmark number.

* The first measurement of retrieval (tool) intent stabilization — distinct from prior incremental SLU, which targets a closed intent ontology — distributed across QA question types and converted into a model-agnostic latency-saving bound.

* A testable cross-linguistic prediction (H5): stabilization, and therefore the payoff of streaming tool use, varies with word order and the locus of grammatical cues.

* An open, CPU-reproducible measurement harness released with the study.

# **10\. Threats to Validity**

* **Construct (input timing).** A uniform word stream is a proxy for speech; real ASR pacing is variable and partial hypotheses get revised. We treat speech timing and ASR-revision effects as a planned follow-up and bound their scope here by using clean text.

* **Construct (stabilization measure).** Top-1 equality is a coarse proxy for “intent settled.” Mitigated by also reporting sufficiency against the gold document and varying top-k.

* **Internal (retriever sensitivity).** BM25 is lexical and phrasing-sensitive; the dense-retriever condition tests whether conclusions are retriever-specific (H4).

* **External (single benchmark).** Results may not transfer beyond CRAG; a second QA set will be added if time permits.

* **Construct (cross-linguistic, H5).** Comparing English CRAG against a different-language benchmark risks confounding language with corpus and topic. We mitigate by using natively authored data rather than machine translation, and by comparing within-language stabilization distributions rather than only cross-language point estimates.

# **11\. Timeline (6 weeks, part-time)**

| Week | Milestone |
| :---- | :---- |
| 1 | Load CRAG; build sparse and dense indices; instrument prefix-streaming retrieval. |
| 2 | Compute stabilization metrics; produce the φ distribution (RQ1). |
| 3 | Sweep (L, δ, θ); streamable-fraction surfaces (RQ2); validate against the harness (RQ3). |
| 4 | Query-feature regression and per-type breakdown (RQ4); retriever and top-k robustness. |
| 5–6 | Write-up; release harness; pilot the SOV cross-linguistic extension (H5) if data permits; target an IR/NLP efficiency workshop or short-paper track. |

# **12\. Resources**

The study is CPU-only and requires no model training: it consists of repeated retrieval over prefixes plus lightweight bookkeeping. Sparse and dense retrieval, the streaming harness, and all analysis run comfortably on a commodity multi-core CPU with \~16 GB RAM. An optional large-language-model reflector (via API) can be added in the RQ3 validation as a robustness check, but is not required for the core result.

# **13\. Anticipated Venue Fit**

The contribution is measurement-and-analysis rather than a new system, which suits efficiency- or retrieval-focused workshops and short-paper tracks (e.g., SIGIR short papers; ACL/EMNLP Findings or associated workshops on efficient or conversational NLP). The released harness also supports a reproducibility-track submission.

# **References**

Altmann, G. T. M., & Kamide, Y. (1999). Incremental Interpretation at Verbs: Restricting the Domain of Subsequent Reference. Cognition, 73(3), 247–264.

Arora, S., Khan, H., Sun, K., Dong, X. L., Choudhary, S., Moon, S., Zhang, X., Sagar, A., Appini, S. T., Patnaik, K., Sharma, S., Watanabe, S., Kumar, A., Aly, A., Liu, Y., Metze, F., & Lin, Z. (2025). Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage. arXiv:2510.02044.

DeVault, D., Sagae, K., & Traum, D. (2011). Incremental Interpretation and Prediction of Utterance Meaning for Interactive Dialogue. Dialogue & Discourse, 2(1).

Grissom II, A., He, H., Boyd-Graber, J., Morgan, J., & Daumé III, H. (2014). Don't Until the Final Verb Wait: Reinforcement Learning for Simultaneous Machine Translation. Proceedings of EMNLP.

He, H., Boyd-Graber, J., & Daumé III, H. (2016). Interpretese vs. Translationese: The Uniqueness of Human Strategies in Simultaneous Interpretation. Proceedings of NAACL-HLT.

Kamide, Y., Altmann, G. T. M., & Haywood, S. L. (2003). The Time-Course of Prediction in Incremental Sentence Processing: Evidence from Anticipatory Eye Movements. Journal of Memory and Language, 49(1), 133–156.

Kamide, Y., & Mitchell, D. C. (1999). Incremental Pre-Head Attachment in Japanese Parsing. Language and Cognitive Processes, 14(5–6), 631–662.

Karpukhin, V., Oğuz, B., Min, S., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. Proceedings of EMNLP.

Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. Proceedings of ICML.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems.

Ma, M., Huang, L., Xiong, H., Zheng, R., Liu, K., Zheng, B., Zhang, C., He, Z., Liu, H., Li, X., Wu, H., & Wang, H. (2019). STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework. Proceedings of ACL.

Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4).

Schlangen, D., & Skantze, G. (2011). A General, Abstract Model of Incremental Dialogue Processing. Dialogue & Discourse, 2(1).

Yang, X., Sun, K., et al. (2024). CRAG: Comprehensive RAG Benchmark. Advances in Neural Information Processing Systems, Datasets and Benchmarks Track.