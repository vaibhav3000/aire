# AIRE Design FAQ

Questions and answers about the design of AIRE, written for code reviews and design discussions. Every claim here matches the code in `src/aire/` and the committed results in `results/`.

## Evaluation fundamentals

### Q1. What is the difference between evaluation and monitoring?

Monitoring answers "is the system healthy right now in production": error rates, latency percentiles, drift signals. Evaluation answers "is this version of the system good enough, and did the change I just made make it better or worse". AIRE is an evaluation tool. It runs a system version against a fixed dataset before release, records traces, and produces comparable metrics. Monitoring observes whatever traffic happens to arrive; evaluation controls the inputs so results are comparable across versions. The two share vocabulary (both use traces and spans) but they answer different questions at different points in the lifecycle.

### Q2. What is the difference between offline and online evaluation?

<!-- aire:begin faq-offline -->
Offline evaluation runs against a curated, versioned dataset with declared expectations (`EvalCase` in `src/aire/runner/dataset.py`). It is reproducible, cheap to re-run, and comparable across versions, but it is only as good as the dataset. Online evaluation judges live traffic, which covers the long tail of real inputs but cannot be replayed exactly and usually relies on sampled human review or proxies. AIRE implements offline evaluation deliberately: the demo's 150-case JSONL set is fixed, hashed into the run manifest (`dataset_sha256`), and identical for every version, which is what makes the regression reports meaningful.
<!-- aire:end faq-offline -->

### Q3. Why evaluate recorded traces instead of scoring inside the application?

Because it decouples "running the system" from "judging the system". In AIRE the runner writes traces once (`runs/<run_id>/traces.jsonl`) and the `EvaluationEngine` reads them (`src/aire/evaluator/engine.py`), never re-running the SUT. Consequences: metrics can be added or changed and applied to old runs without API cost; evaluation is debuggable because the exact input and output of every score is on disk; and evaluation cannot perturb the system it measures. This is the "run once, evaluate many times" property stated in the engine docstring.

### Q4. Reference-based vs reference-free evaluation: which does AIRE use?

Both, selected per case. Reference-based metrics compare the output to a declared expectation: `expected_keywords` (keyword_recall), `expected_answer` (exact_match), `supporting_doc_ids` (retrieval metrics and gold_citation_recall). Reference-free metrics check internal consistency without a gold answer: groundedness checks that each claim is supported by a cited document the system retrieved, and citation_precision checks that what the answer cites is either a supporting or a retrieved document. Abstention checking is reference-free in a different sense: for unanswerable questions the expectation is a behavior (refuse), not a string. The engine only computes the metrics a case actually declares (see `evaluate_case` in `engine.py`).

### Q5. What is groundedness in AIRE, and what are the limits of the lexical proxy?

Definition (`src/aire/evaluator/metrics.py`, `groundedness`): split the answer into claims at sentence terminators and line breaks, so every sentence and every bullet needs its own citation. Citations written right after a sentence's period belong to that sentence ("S1. [doc:x] S2." cites S1), following the convention that a citation follows the claim it supports. For each claim, compare its content keywords with those of the documents it cites, using only documents the system actually retrieved: the engine passes the retrieved documents as the context, so citing a document the system never saw cannot make a claim grounded. A claim counts as supported when at least `EngineConfig.groundedness_min_overlap` (default 0.5) of its keywords appear in such a document. The score is supported claims divided by claims with content keywords. Uncited claims count as unsupported; empty or abstaining answers return `None` (not measurable), and a failed invocation scores 0.

<!-- aire:begin faq-citation-bug -->
An earlier version split only after periods and therefore attached a citation written after the period to the next sentence. Extractive answers in the "S1. [doc:x] S2. [doc:x]" form then scored 0.5 although they quote the cited document verbatim; that bug produced the old 0.5682 baseline. `tests/test_metrics.py` now checks that verbatim extractive output scores 1.0 for both citation placements.
<!-- aire:end faq-citation-bug -->

Limits: this is a lexical citation-overlap proxy, not entailment. It is blind to paraphrase in one direction and fooled by vocabulary in the other: a sentence that rewords the source with different words scores as unsupported even when it is faithful, and at the default threshold a sentence whose keywords are half from the source and half invented scores as supported. It verifies citation hygiene and copy fidelity, not truth. The docstring says this explicitly, and the `GroundednessJudge` returns a detail string labelling itself "lexical citation-overlap proxy; deterministic" so the number cannot be mistaken for a semantic judgment.

### Q6. Deterministic metrics vs LLM-as-judge: when do you use which?

Deterministic metrics (`metrics.py`, `retrieval_metrics.py`, `tool_metrics.py`) are pure functions of the trace and case: no network, no model calls, exactly reproducible, unit-testable in microseconds. Use them whenever the expectation can be declared structurally: keywords present, docs retrieved, tool called, citations attached. LLM-as-judge is for dimensions with no structural ground truth: helpfulness, tone, overall answer quality. AIRE ships one built-in deterministic judge (`GroundednessJudge`) and a `LLMJudge` interface (`src/aire/evaluator/judges.py`) for model-based scoring. The interface deliberately records the judge's description and version in every result and attaches the warning "model-based judge; validate against human labels before trusting". No committed demo result uses an LLM judge.

### Q7. What biases does LLM-as-judge have, and how does AIRE position itself?

The known biases: position bias (grading depends on where an answer appears in a comparison), verbosity bias (longer answers score higher regardless of quality), and self-preference (a model rates its own outputs higher than other models' outputs, and its own style as more correct). Because of these, an LLM judge is a measurement instrument that itself needs calibration: agreement with human labels must be measured before the scores are trusted. AIRE's position, stated in `judges.py`: do not oversell the judge. The `LLMJudge` wrapper records judge identity and version in the result detail, rejects out-of-range scores, and demands human-label validation as the integrator's responsibility. Scores without provenance metadata are not auditable, so provenance is mandatory.

### Q8. Why is the demo system under test deterministic?

Reproducibility of committed results. Every number in the README, the JSON reports, and the HTML dashboards was produced by `ExtractiveRAG`, a rule-based keyword retriever plus extractive responder. Because it has no sampling and no network, anyone can re-run `python demo/run_demo.py` and get identical metrics; only ids, timestamps and latencies differ, and the script leaves the committed files untouched when nothing else changed, which is what lets CI fail on any drift of the committed artifacts. That makes the evaluation framework itself testable and reviewable: the interesting artifact of this project is the evaluator, and the evaluator's outputs must be checkable. The `SystemUnderTest` interface accepts real LLM adapters unchanged (implementations declare `deterministic = False` when they are not, and the manifest records it), so determinism is a property of the demo choice, not of the framework.

## Metrics in depth

### Q9. How do the retrieval metrics work, and what is the ground truth?

<!-- aire:begin faq-retrieval -->
`retrieval_recall_at_k` is the fraction of relevant documents found within the first k retrieved; `retrieval_mrr` is the reciprocal rank of the first relevant document within the first k, 0 if none. The engine reports them at fixed cut-offs (`EngineConfig.recall_ks`, default 1 and 3, and `mrr_k`, default 3) as `retrieval_recall_at_1`, `retrieval_recall_at_3` and `retrieval_mrr_at_3`. A fixed K matters: an un-cut recall compares a system that retrieves one document with one that retrieves three, which is how the old report credited idf weighting with a gain that came from K. Repeated ids count once, and a case with an empty retrieval scores 0 instead of dropping out of the mean. Ground truth is declarative: each `EvalCase` lists `supporting_doc_ids`, the documents that should be retrieved for that input. `retrieval_metrics.py` is short on purpose. The honest difficulty in retrieval evaluation is constructing reliable ground truth for real corpora, and the module docstring says the project does not pretend to have solved that. For the demo, the 12-document corpus makes hand-labelling feasible.
<!-- aire:end faq-retrieval -->

### Q10. How are tool calls evaluated?

Three questions, in `tool_metrics.py`: (1) `tool_selection_correct`: did the system call the expected tool at all (extra calls are not penalized here; that is what traces are for). (2) `tool_args_valid`: fraction of recorded calls whose arguments passed schema validation, where each recorded call carries `{"name", "args", "valid"}`; a call the runtime rejected for a schema violation still counts against the system, because a malformed call that never executed is still the system's failure. (3) `tool_args_match`: fraction of expected argument key/value pairs present in the first call of the expected tool. All three return `None` when the case declares no tool expectation, so the engine reports only applicable metrics.

### Q11. How does cost accounting work, given the demo costs nothing?

The trace records `Usage` with `prompt_tokens`, `completion_tokens`, and `token_source`. The engine computes `cost_usd = total_tokens / 1000 * price_per_1k` (`metrics.cost_estimate`), with the price coming from `EngineConfig`. The demo price table is 0.0 USD per 1k tokens because the token counts themselves are heuristic (see Q12): charging money for estimated tokens would produce a fake number. The design point is that cost accounting is a pure function of recorded usage plus a price table, so attaching real API usage and real prices later requires no evaluator changes.

### Q12. Why are heuristic token counts labelled?

Because provenance determines how a number may be used. When the SUT does not report real usage, the runner estimates tokens at about 4 characters per token (`schema.estimate_tokens`) and writes `token_source: "heuristic"` into the `Usage` record. A failed call reports nothing and gets `token_source: "none"` with zero tokens; it is never filled with an estimate. Any downstream consumer can therefore distinguish measured from estimated counts: token means skip `"none"` rows and failed cases, the aggregates list the sources they mixed (`token_sources`), and the regression engine gives no token verdict when two runs used different sources. The alternative, silently mixing real and estimated counts under one field, is how misleading cost dashboards get built (and how the old LLM comparison ended up setting API counts against a character estimate). The `Usage` docstring states the rule: heuristic counters are allowed but must be labelled.

## Regression detection

### Q13. How does the regression engine decide that something regressed?

`compare_runs` in `src/aire/regression/compare.py` first checks that the two runs are comparable: when both have manifests their `dataset_sha256` must match, and the case ids must match in every case; otherwise it raises `RunMismatchError` and the gate exits 2. It then compares every `mean_<metric>` aggregate and `success_rate`. Direction-aware: the metric registry (`metrics.metric_spec`) declares whether higher is better (quality, retrieval, tool metrics, success rate) or lower is better (latency, cost, tokens, errors, false refusals). Tolerance-gated: a change counts as a verdict only beyond the metric's tolerance (default 0.02). A metric present in the baseline but missing from the candidate is a regression, so a candidate cannot pass by producing fewer metrics, and a mean over a different number of cases is flagged with both counts.

### Q14. Why absolute tolerances for scores and relative ones for cost?

Scores in [0, 1] (groundedness, recall, keyword recall, success rate) are already on a common scale, so the tolerance is an absolute difference: 0.02 means two points whatever the baseline. A relative rule misbehaves there: a move from 0.10 to 0.085 is -15 percent but only 1.5 points, and any change from a zero baseline is an infinite relative change (the old engine wrote a relative delta of 1e9 into a report). Latency, tokens and cost have no fixed scale, so for them the tolerance is a relative change against the baseline, falling back to an absolute difference when the baseline is zero. Per-metric overrides handle the exceptions (see Q15), and per-case attribution applies the same rule as the aggregate.

### Q15. What do latency threshold overrides represent?

Encoded team judgment. Thresholds are not universal constants; they are policy about what magnitude of change matters for that metric in that context, and `--override latency_ms=0.5` (or `RegressionThresholds.overrides`) is how that policy gets written down and versioned with the experiment code. For the offline demo the policy is stronger: sub-millisecond latencies are noise on any machine, so `run_demo.py` and the CI gate report latency but leave it out of the verdict counts (`ignore={"latency_ms"}` and `--ignore latency_ms`). The old demo gated latency at 10 percent, and its committed summary (3 regressed, 3 improved for v3) flipped on a rerun because of timing jitter alone.

### Q16. What is per-case attribution and why does every regression need it?

<!-- aire:begin faq-attribution -->
An aggregate regression without attribution is not actionable. When a metric is marked regressed, `compare_runs` attaches the list of individual cases whose scores moved beyond the tolerance on that metric (or lost it entirely), with baseline value, candidate value, and delta per case, sorted worst first (`_attribute_regressions`). The aggregate tells you groundedness collapsed; the attribution tells you it collapsed on the 130 answerable cases where the fluent style dropped citations. Per-case rows also carry a `failure` label, so the failure-count deltas (for example `ungrounded` +109) line up with the metric attribution.
<!-- aire:end faq-attribution -->

## Design decisions

### Q17. Walk through the trace schema. Why spans and why a `token_source` label?

A `Trace` (`src/aire/schema.py`) is one invocation for one case: identity (`run_id`, `case_id`, `trace_id`), the system config fingerprint, `final_answer`, `retrieved_doc_ids`, `tool_calls`, an ordered list of `Span` objects, a `Usage` record, and an optional top-level `error`. Spans carry kind (`retrieval`, `generation`, `tool_call`, `custom`), name, wall-clock timing, `latency_ms`, arbitrary attributes, and a per-span error, so multi-step systems can be instrumented with `TraceRecorder.span(...)` context managers. The runner writes one span per attempt, failed attempts included, and records deliberate waits reported by the system (`wait_ms`, such as rate-limit backoff between provider retries) as an attribute instead of counting them as latency; client-side pacing runs in `before_invoke()`, before the timer starts. `token_source` exists because of provenance (Q12): `"api"` for counts reported by a real backend, `"heuristic"` for the ~4-chars-per-token estimate, `"none"` when unknown. Every serialised trace carries `schema_version`. The full schema with a worked example is in `docs/trace_schema.md`.

### Q18. Explain the failure taxonomy. Why first-match-wins, and why this order?

`classify_failure` in `engine.py` maps each case result to exactly one category, checked in pipeline order: `invocation_error` (the call failed or returned no answer), `missed_abstention` (answered an unanswerable question), `retrieval_miss` (a gold document is missing from the top three retrieved), `tool_error` (wrong tool, wrong arguments or an invalid call), `wrong_abstention` (refused although the gold documents were retrieved), `ungrounded` (groundedness below 0.5), `incomplete_answer` (keyword recall below 0.5), `answer_mismatch` (an exact-match expectation was missed), or none (success). First-match-wins makes the categories mutually exclusive, so failure counts sum to something meaningful and a single case does not inflate four counters at once. Upstream stages come first so a failure is blamed on its earliest cause: an answer built without the gold document is a retrieval problem, not a generation problem, and a model that refuses because retrieval brought nothing relevant did the right thing given its context. The earlier order checked retrieval last, so five of v1's nine `incomplete_answer` cases were really retrieval misses. The categories are exclusive by construction, not exhaustive: they cover the failure modes the metrics measure.

### Q19. Why not build on DeepEval or Ragas?

The evaluator is the project. AIRE's substance is the trace schema, the metric definitions with their honest docstrings, the failure taxonomy, the run storage format, the regression engine with direction-aware tolerances, run-identity checks and attribution, and the judge interface with provenance requirements. Wrapping DeepEval or Ragas would outsource exactly that substance and inherit their scoring definitions, which AIRE would then be unable to explain line by line. Building the metrics from first principles also keeps the zero-dependency guarantee (`pyproject.toml` has an empty `dependencies` list), which is what makes offline, reproducible evaluation trivial to install. A framework can still be added at the edges (for example an adapter exposing a Ragas metric as a `Judge`), but the core definitions stay owned and auditable.

### Q20. What does the abstention finding show about the value of an evaluator?

<!-- aire:begin faq-abstention -->
The demo's `abstention_correct` is 0.0 for v1 and 0.1 for v2 and v3: of the 20 unanswerable questions, v1 refuses none and v2 two. The measurement behind that failure is in the README: with idf weighting, unanswerable queries produce top retrieval scores between 0.0056 and 0.4596, while 84 of the 130 answerable queries score no higher than that, so any `min_score` that rejects every unanswerable query also rejects most answerable ones; fixing it requires semantic matching, not threshold tuning. The point is what the evaluator did: it converted a vague worry ("does the system hallucinate on out-of-domain questions?") into a measured capability gap with the score distributions to prove it, and the regression reports carried that gap across versions instead of burying it under averages that looked acceptable. The metric rests on 20 cases, which is enough to show the gap but not to measure a small improvement; `abstention_accuracy` and `false_refusal` cover the answerable side.
<!-- aire:end faq-abstention -->

### Q21. What do the v2 and v3 demo comparisons demonstrate?

<!-- aire:begin faq-comparisons -->
v2 vs v1 shows a clean change with an honest explanation: retrieving three documents instead of one lifted `retrieval_recall_at_3` from 0.7154 to 0.8615 and `retrieval_mrr_at_3` from 0.7769 to 0.8231, both significant in the paired bootstrap, with no regressions detected (2 improved, 10 neutral, 2 inconclusive). The success rate rose from 0.4267 to 0.4667, but McNemar's test on the discordant cases gives p = 0.07, so the gate reports it as inconclusive instead of crediting it. `retrieval_recall_at_1` did not move, and the ablation in `results/retrieval_ablation.json` shows that at equal K plain overlap weighting is at least as good as idf on this corpus (recall@3 0.8731 vs 0.8615), so the gain is K, not idf. v3 vs v1 shows the interesting case: the fluent style keeps v2's retrieval but drops every citation, so `groundedness` falls from 1.0 to 0.0, `gold_citation_recall` from 0.7154 to 0.0 and `success_rate` from 0.4267 to 0.0133, `citation_precision` disappears, and `ungrounded` failures rise by 109. The regression report surfaces the retrieval-versus-grounding trade-off as separate rows instead of hiding it behind a single score: a composite might have netted v3 out as "fine".
<!-- aire:end faq-comparisons -->

### Q22. How does validation work, and when does it run?

`validate_trace_dict` checks structure: required identity fields (`run_id`, `case_id`, `trace_id`), every span has `kind` and `latency_ms`, no negative latencies, no negative token counts, and every tool call is an object with a name. It runs on write: `RunStore.append_trace` raises on an invalid trace or a second trace for the same case, and `TraceRecorder.finish` validates before returning, so anything stored in `traces.jsonl` is already trusted. The dataset loader is strict too: unknown fields (a typo such as `expect_abstension`), non-boolean flags, malformed lists and duplicate case ids raise with the line number instead of being dropped or passed through. Fail-loud-at-the-boundary was chosen because corrupt data written once propagates into every downstream metric silently, whereas a write-time exception stops the experiment immediately. Readers, in turn, ignore trace fields they do not know, so additive schema changes stay readable.

### Q23. How would you attach a real LLM system?

Implement `SystemUnderTest` (one method, `invoke(case_input, case_context) -> SUTResult`), set `deterministic = False` so the manifest records that replays are not bit-identical, and hand the instance to the same `ExperimentRunner`. Put client-side pacing in `before_invoke()`, which the runner calls before it starts the timer, and report deliberate waits inside `invoke()` as `SUTResult.wait_ms` so they stay out of latency. The runner injects the SUT and never imports its dependencies, so the OpenAI-compatible adapter needs no changes to the runner, engine, metrics, or regression code. Provide real usage counts in the result when the API reports them so `token_source` can be `"api"`, set a real price table in `EngineConfig`, and expect the deterministic metrics (citations, keywords, retrieval) to work unchanged while judgment-dependent dimensions move to an `LLMJudge`, validated against human labels per Q7.

### Q24. What are the current limitations you would state to a reviewer?

<!-- aire:begin faq-limitations -->
From the README's limitations section, in descending order of importance: lexical groundedness is a citation-hygiene proxy that paraphrase can defeat, not entailment; the gate's paired tests and intervals treat the 150 cases as a sample of questions but say nothing about run-to-run model noise, which needs repeated runs; the demo SUT is a deterministic stand-in, so committed latency and token figures characterize the harness, not an LLM; token counts are heuristic and labelled as such; abstention is a measured open gap requiring semantic matching (2 of 20 unanswerable questions refused by v2); retrieval ground truth is hand-declared for a 12-document corpus and does not scale as-is; and the sync runner's thread-based timeout bounds its wait but cannot kill a hung call in Python (the async runner cancels it). Each limitation is documented where the relevant code lives rather than in a disclaimer nobody reads.
<!-- aire:end faq-limitations -->

## Deterministic vs real-LLM mode

**Why evaluate both a deterministic responder and a real LLM on the same suite?**
The deterministic responder is a calibration harness: every number is
reproducible, so evaluator bugs show up as metric changes instead of being
hidden inside LLM noise. The citation-placement bug in groundedness (Q5) is
the example: verbatim extractive quotes scored 0.5, which is impossible for a
correct metric and points straight at the metric. The real-LLM run then shows
what the evaluator says about a genuine model. The two are stored separately
and compared only through the regression engine, which labels every delta.

**What did the real LLM change on this suite?**
Not much that can still be claimed. The committed run (gemini-2.5-flash,
committed on 2026-09-17) was measured once with the metric code from before
the fixes in this repository, and its traces were not kept, so it cannot be
<!-- aire:begin faq-llm-history -->
re-scored. It answered "I don't know." on all four unanswerable questions,
where both deterministic versions answer, which is a real but four-case
observation. Its groundedness of 1.0 averaged only the 15 cases that made a
claim, against a deterministic baseline that the citation-placement bug had
pushed down to 0.5682, so there was no measured grounding gain. Its 15.3 s
<!-- aire:end faq-llm-history -->
mean latency included the client's own pacing and retry sleeps, and its token
mean mixed API counts with estimates. `scripts/run_llm_eval.py` now persists
traces and can re-score them (`--rescore`), so the next run will be
replayable.

**What does the timeout incident demonstrate?**
The first LLM sweep lost 14 of 26 cases to the runner's 30-second invocation
watchdog (thinking models generate for 30-90s). The failure taxonomy classed
them as `invocation_error`, correctly, since the system under test failed to
answer in time, whatever the cause. The lesson is that reliability
configuration is part of the system under test: the LLM script now derives the
runner timeout from the provider's worst-case call duration (attempts times
request timeout plus capped backoff), and the provider retries only transient
failures, so a quota error fails fast instead of burning the timeout.
