# AIRE Interview Guide

Questions and answers about the design of AIRE, written for code reviews, design discussions, and interview preparation. Every claim here matches the code in `src/aire/` and the committed results in `results/`.

## Evaluation fundamentals

### Q1. What is the difference between evaluation and monitoring?

Monitoring answers "is the system healthy right now in production": error rates, latency percentiles, drift signals. Evaluation answers "is this version of the system good enough, and did the change I just made make it better or worse". AIRE is an evaluation tool. It runs a system version against a fixed dataset before release, records traces, and produces comparable metrics. Monitoring observes whatever traffic happens to arrive; evaluation controls the inputs so results are comparable across versions. The two share vocabulary (both use traces and spans) but they answer different questions at different points in the lifecycle.

### Q2. What is the difference between offline and online evaluation?

Offline evaluation runs against a curated, versioned dataset with declared expectations (`EvalCase` in `src/aire/runner/dataset.py`). It is reproducible, cheap to re-run, and comparable across versions, but it is only as good as the dataset. Online evaluation judges live traffic, which covers the long tail of real inputs but cannot be replayed exactly and usually relies on sampled human review or proxies. AIRE implements offline evaluation deliberately: the demo's 26-case JSONL set is fixed, hashed into the run manifest (`dataset_sha256`), and identical for every version, which is what makes the regression reports meaningful.

### Q3. Why evaluate recorded traces instead of scoring inside the application?

Because it decouples "running the system" from "judging the system". In AIRE the runner writes traces once (`runs/<run_id>/traces.jsonl`) and the `EvaluationEngine` reads them (`src/aire/evaluator/engine.py`), never re-running the SUT. Consequences: metrics can be added or changed and applied to old runs without API cost; evaluation is debuggable because the exact input and output of every score is on disk; and evaluation cannot perturb the system it measures. This is the "run once, evaluate many times" property stated in the engine docstring.

### Q4. Reference-based vs reference-free evaluation: which does AIRE use?

Both, selected per case. Reference-based metrics compare the output to a declared expectation: `expected_keywords` (keyword_recall), `expected_answer` (exact_match), `supporting_doc_ids` (retrieval metrics). Reference-free metrics check internal consistency without a gold answer: groundedness checks that cited sentences are actually supported by the cited documents, and citation_coverage checks that supporting material is referenced. Abstention checking is reference-free in a different sense: for unanswerable questions the expectation is a behavior (refuse), not a string. The engine only computes the metrics a case actually declares (see `evaluate_case` in `engine.py`).

### Q5. What is groundedness in AIRE, and what are the limits of the lexical proxy?

Definition (`src/aire/evaluator/metrics.py`, `groundedness`): split the answer into sentences, and for each sentence that carries a `[doc:<id>]` citation, compute the overlap between the sentence's content keywords and the cited document's content keywords. A sentence counts as supported when at least 50 percent of its keywords appear in the document. The score is supported cited sentences divided by checked sentences. Uncited factual sentences count as unsupported; empty or abstaining answers return `None` (not measurable).

Limits: this is a lexical citation-overlap proxy, not entailment. It is blind to paraphrase in one direction and fooled by vocabulary in the other: a sentence that rewords the source with different words scores as unsupported even when it is faithful, and a fluent nonsense sentence that reuses source vocabulary can score as supported. It verifies citation hygiene and copy fidelity, not truth. The docstring says this explicitly, and the `GroundednessJudge` returns a detail string labelling itself "lexical citation-overlap proxy; deterministic" so the number cannot be mistaken for a semantic judgment.

### Q6. Deterministic metrics vs LLM-as-judge: when do you use which?

Deterministic metrics (`metrics.py`, `retrieval_metrics.py`, `tool_metrics.py`) are pure functions of the trace and case: no network, no model calls, exactly reproducible, unit-testable in microseconds. Use them whenever the expectation can be declared structurally: keywords present, docs retrieved, tool called, citations attached. LLM-as-judge is for dimensions with no structural ground truth: helpfulness, tone, overall answer quality. AIRE ships one built-in deterministic judge (`GroundednessJudge`) and a `LLMJudge` interface (`src/aire/evaluator/judges.py`) for model-based scoring. The interface deliberately records the judge's description and version in every result and attaches the warning "model-based judge; validate against human labels before trusting". No committed demo result uses an LLM judge.

### Q7. What biases does LLM-as-judge have, and how does AIRE position itself?

The known biases: position bias (grading depends on where an answer appears in a comparison), verbosity bias (longer answers score higher regardless of quality), and self-preference (a model rates its own outputs higher than other models' outputs, and its own style as more correct). Because of these, an LLM judge is a measurement instrument that itself needs calibration: agreement with human labels must be measured before the scores are trusted. AIRE's position, stated in `judges.py`: do not oversell the judge. The `LLMJudge` wrapper records judge identity and version in the result detail, rejects out-of-range scores, and demands human-label validation as the integrator's responsibility. Scores without provenance metadata are not auditable, so provenance is mandatory.

### Q8. Why is the demo system under test deterministic?

Reproducibility of committed results. Every number in the README, the JSON reports, and the HTML dashboards was produced by `ExtractiveRAG`, a rule-based keyword retriever plus extractive responder. Because it has no sampling and no network, anyone can re-run `python demo/run_demo.py` and get identical metrics (up to timing jitter in latency). That makes the evaluation framework itself testable and reviewable: the interesting artifact of this project is the evaluator, and the evaluator's outputs must be checkable. The `SystemUnderTest` interface accepts real LLM adapters unchanged (see `llm/base.py`: implementations declare `deterministic = False` when they are not), so determinism is a property of the demo choice, not of the framework.

## Metrics in depth

### Q9. How do the retrieval metrics work, and what is the ground truth?

`retrieval_recall_at_k` is the fraction of relevant documents found within the first k retrieved; `retrieval_mrr` is the reciprocal rank of the first relevant document, 0 if none. Ground truth is declarative: each `EvalCase` lists `supporting_doc_ids`, the documents that should be retrieved for that input. `retrieval_metrics.py` is short on purpose. The honest difficulty in retrieval evaluation is constructing reliable ground truth for real corpora, and the module docstring says the project does not pretend to have solved that. For the demo, the 12-document corpus makes hand-labelling feasible.

### Q10. How are tool calls evaluated?

Three questions, in `tool_metrics.py`: (1) `tool_selection_correct`: did the system call the expected tool at all (extra calls are not penalized here; that is what traces are for). (2) `tool_args_valid`: fraction of recorded calls whose arguments passed schema validation, where each recorded call carries `{"name", "args", "valid"}`; a call the runtime rejected for a schema violation still counts against the system, because a malformed call that never executed is still the system's failure. (3) `tool_args_match`: fraction of expected argument key/value pairs present in the first call of the expected tool. All three return `None` when the case declares no tool expectation, so the engine reports only applicable metrics.

### Q11. How does cost accounting work, given the demo costs nothing?

The trace records `Usage` with `prompt_tokens`, `completion_tokens`, and `token_source`. The engine computes `cost_usd = total_tokens / 1000 * price_per_1k` (`metrics.cost_estimate`), with the price coming from `EngineConfig`. The demo price table is 0.0 USD per 1k tokens because the token counts themselves are heuristic (see Q12): charging money for estimated tokens would produce a fake number. The design point is that cost accounting is a pure function of recorded usage plus a price table, so attaching real API usage and real prices later requires no evaluator changes.

### Q12. Why are heuristic token counts labelled?

Because provenance determines how a number may be used. When the SUT does not report real usage, the runner estimates tokens at about 4 characters per token (`schema.estimate_tokens`) and writes `token_source: "heuristic"` into the `Usage` record. Any downstream consumer (aggregation, cost, reports) can therefore distinguish measured from estimated counts. The alternative, silently mixing real and estimated counts under one field, is how misleading cost dashboards get built. The `Usage` docstring states the rule: heuristic counters are allowed but must be labelled.

## Regression detection

### Q13. How does the regression engine decide that something regressed?

`compare_runs` in `src/aire/regression/compare.py` compares every aggregate metric present in both runs. Three properties matter. Direction-aware: the metric registry (`metrics.metric_spec`) declares whether higher is better (quality, retrieval, tool metrics) or lower is better (latency, cost, tokens, errors), so a latency increase is a regression while a groundedness increase is an improvement, mechanically. Relative: the delta is divided by the baseline value, so thresholds are scale-free. Threshold-gated: a delta counts as a verdict only when it exceeds the metric's threshold (default 2 percent relative), which keeps numerical noise out of the report.

### Q14. Why relative thresholds instead of absolute deltas?

Because absolute deltas mean different things at different scales. The comment in `compare.py` gives the real example: a +0.58 token change on a mean of about 83 tokens is noise, while a -0.57 change in groundedness from a baseline of 0.5682 is a catastrophe. Both are roughly the same absolute size. Relative change makes one default threshold (2 percent) meaningful across metrics of wildly different magnitudes, and per-metric overrides handle the exceptions (see Q15).

### Q15. What do latency threshold overrides represent?

Encoded team judgment. The demo passes `RegressionThresholds(default=0.02, overrides={"latency_ms": 0.10})`. The comment in `run_demo.py` explains: latency differences below 10 percent are noise at sub-millisecond scale, so a 2 percent default would flag meaningless jitter (and did: v2's +3.6 percent latency delta is correctly reported as neutral under the override). Thresholds are not universal constants; they are policy about what magnitude of change matters for that metric in that context. Letting callers override them per metric is how that policy gets written down and versioned with the experiment code instead of living in someone's head.

### Q16. What is per-case attribution and why does every regression need it?

An aggregate regression without attribution is not actionable. When a metric is marked regressed, `compare_runs` attaches the list of individual cases whose scores moved beyond the threshold on that metric, with baseline value, candidate value, and delta per case, sorted worst first (`_attribute_regressions`). The aggregate tells you groundedness collapsed; the attribution tells you it collapsed on the 22 cases where the fluent style dropped citations. Per-case rows also carry a `failure` label, so the failure-count deltas (for example `ungrounded` +22) line up with the metric attribution.

## Design decisions

### Q17. Walk through the trace schema. Why spans and why a `token_source` label?

A `Trace` (`src/aire/schema.py`) is one invocation for one case: identity (`run_id`, `case_id`, `trace_id`), the system config fingerprint, `final_answer`, `retrieved_doc_ids`, `tool_calls`, an ordered list of `Span` objects, a `Usage` record, and an optional top-level `error`. Spans carry kind (`retrieval`, `generation`, `tool_call`, `custom`), name, wall-clock timing, `latency_ms`, arbitrary attributes, and a per-span error, so multi-step systems can be instrumented with `TraceRecorder.span(...)` context managers. `token_source` exists because of provenance (Q12): `"api"` for counts reported by a real backend, `"heuristic"` for the ~4-chars-per-token estimate, `"none"` when unknown. The full schema with a worked example is in `docs/trace_schema.md`.

### Q18. Explain the failure taxonomy. Why first-match-wins?

`classify_failure` in `engine.py` maps each case result to exactly one category from a small ordered list: `invocation_error` (the call itself failed), `wrong_abstention` (refused an answerable question), `missed_abstention` (answered an unanswerable one), `ungrounded` (groundedness below 0.5), `incomplete_answer` (keyword recall below 0.5), `retrieval_miss` (supporting docs not fully retrieved), or none (success). First-match-wins makes the categories mutually exclusive, so failure counts sum to something meaningful and a single case does not inflate four counters at once. The ordering encodes causality: if the invocation errored there is nothing else to say; abstention errors dominate grounding because an abstention is a different behavior class, not a bad answer. A coarse taxonomy that is consistent beats a fine one that double-counts.

### Q19. Why not build on DeepEval or Ragas?

The evaluator is the project. AIRE's substance is the trace schema, the metric definitions with their honest docstrings, the failure taxonomy, the run storage format, the regression engine with direction-aware relative thresholds and attribution, and the judge interface with provenance requirements. Wrapping DeepEval or Ragas would outsource exactly that substance and inherit their scoring definitions, which AIRE would then be unable to explain line by line. Building the metrics from first principles also keeps the zero-dependency guarantee (`pyproject.toml` has an empty `dependencies` list), which is what makes offline, reproducible evaluation trivial to install. A framework can still be added at the edges (for example an adapter exposing a Ragas metric as a `Judge`), but the core definitions stay owned and auditable.

### Q20. What does the abstention finding show about the value of an evaluator?

The demo's `abstention_correct` is 0.0 for v1 and v2: the system never refuses an unanswerable question. The measurement behind that failure is in the README: unanswerable queries produce top retrieval scores between 0.18 and 0.46, while the weakest answerable queries score between 0.19 and 0.26. The ranges overlap, so no calibrated `min_score` threshold can separate the two populations; fixing it requires semantic matching, not threshold tuning. The point is what the evaluator did: it converted a vague worry ("does the system hallucinate on out-of-domain questions?") into a measured capability gap with the score distributions to prove it, and the regression reports carried that gap across versions instead of burying it under averages that looked acceptable. An evaluator earns its keep precisely on failures like this one, which a demo built only to look good would hide.

### Q21. What do the v2 and v3 demo comparisons demonstrate?

v2 vs v1 shows a clean improvement: idf weighting plus top-3 retrieval lifted `retrieval_recall` from 0.7273 to 0.9091 and `retrieval_mrr` from 0.7727 to 0.8333 with no regressions detected (2 improved, 7 neutral). v3 vs v1 shows the interesting case: the fluent style improved retrieval metrics further (`retrieval_recall` 0.9545) and made answers read better, but collapsed `citation_coverage` to 0.0 and `groundedness` to 0.0, with `ungrounded` failures up 22. The regression report surfaces the fluency-versus-grounding trade-off as two columns instead of hiding it behind a single score. This is the argument for evaluating many metrics with direction and attribution rather than one composite number: a composite might have netted v3 out as "fine".

### Q22. How does validation work, and when does it run?

`validate_trace_dict` checks structure: required identity fields (`run_id`, `case_id`, `trace_id`), every span has `kind` and `latency_ms`, no negative latencies, no negative token counts. It runs on write, not on read: `RunStore.append_trace` raises on an invalid trace and `TraceRecorder.finish` validates before returning, so anything stored in `traces.jsonl` is already trusted. Fail-loud-at-the-boundary was chosen over schema enforcement at read time because corrupt data written once propagates into every downstream metric silently, whereas a write-time exception stops the experiment immediately.

### Q23. How would you attach a real LLM system?

Implement `SystemUnderTest` (one method, `invoke(case_input, case_context) -> SUTResult`), set `deterministic = False` so the framework knows replays are not bit-identical, and hand the instance to the same `ExperimentRunner`. The runner injects the SUT and never imports its dependencies, so the OpenAI-compatible adapter needs no changes to the runner, engine, metrics, or regression code. Provide real usage counts in the result when the API reports them so `token_source` can be `"api"`, set a real price table in `EngineConfig`, and expect the deterministic metrics (citations, keywords, retrieval) to work unchanged while judgment-dependent dimensions move to an `LLMJudge`, validated against human labels per Q7.

### Q24. What are the current limitations you would state to a reviewer?

From the README's limitations section, in descending order of importance: lexical groundedness is a citation-hygiene proxy that paraphrase can defeat, not entailment; the demo SUT is a deterministic stand-in, so committed latency and token figures characterize the harness, not an LLM; token counts are heuristic and labelled as such; abstention is a measured open gap requiring semantic matching; retrieval ground truth is hand-declared for a 12-document corpus and does not scale as-is; and the thread-based timeout bounds the runner's wait but cannot kill a hung call in Python. Each limitation is documented where the relevant code lives rather than in a disclaimer nobody reads.


## Deterministic vs real-LLM mode

**Why evaluate both a deterministic responder and a real LLM on the same suite?**
The deterministic responder is a calibration harness: every number is
reproducible, so evaluator bugs (like the citation-attachment bug found during
development) show up as metric changes instead of being hidden inside LLM
noise. The real-LLM run then shows what the evaluator says about a genuine
model. The two are stored separately (`llm_eval_results.json` vs the
deterministic run artifacts) and compared only through the regression engine,
which labels every delta.

**What did the real LLM change on this suite?**
Measured (2026-09-16, gemini-2.5-flash): perfect abstention on all four
unanswerable questions where both deterministic versions scored zero - a real
capability gain the regression engine flags as improved. Groundedness of its
cited answers was 1.0 under the same lexical judge. The costs were equally
real: lower keyword recall than extractive quoting, seconds of latency per
case, and hundreds of tokens per answer. The evaluator surfaces the
trade-off instead of a one-number verdict.

**What does the timeout incident demonstrate?**
The first LLM sweep lost 14 of 26 cases to the runner's 30-second invocation
watchdog (thinking models generate for 30-90s). The failure taxonomy classed
them as `invocation_error` - correctly, since the SYSTEM under test failed to
answer in time, whatever the cause. Raising the runner timeout and re-running
is the operational lesson: reliability configuration is part of the system
under test, and the evaluator made the failure visible instead of silent.
