# AIRE glossary

AIRE evaluates AI systems (RAG pipelines, LLM applications and tool-using
agents) from recorded traces and decides whether a new version regressed.
These are the words the code, the docs and the reports use.

## Runs and records

**Case**:
One question or task together with the expectations used to score the answer.
_Avoid_: example, sample, test case

**Eval set**:
The fixed list of cases a run is scored on, identified by its content hash.
_Avoid_: dataset (in prose), benchmark

**Corpus**:
The documents a system may retrieve from and that groundedness is checked against.
_Avoid_: knowledge base, context store

**System under test**:
The version of the AI system being evaluated, with its configuration.
_Avoid_: model (a system contains a model), app, pipeline

**Run**:
One system under test executed once over one eval set, stored as a manifest and traces.
_Avoid_: experiment, job, execution

**Trace**:
The complete record of one case in one run: answer, retrieved documents, tool calls, spans and token usage.
_Avoid_: log, transcript

**Span**:
One timed step inside a trace, such as a model attempt, a retrieval or a tool call; a root span covers the whole case.
_Avoid_: event, step (in this sense)

**Token source**:
Where a token count came from: the provider's own count, an estimate, or none.
_Avoid_: token type

## Calling models

**Provider**:
The organisation or server software that serves a model (Gemini, OpenAI, Anthropic, an OpenAI-compatible server).
_Avoid_: vendor, backend

**Adapter**:
The piece that speaks one provider's protocol and makes exactly one attempt per request.
_Avoid_: driver, connector, wrapper

**Model client**:
The shared caller in front of an adapter that paces requests, retries failures and caches responses.
_Avoid_: provider (for this role), wrapper

**Rate budget**:
The requests per minute and tokens per minute a provider allows.
_Avoid_: quota (quota means the daily or billing allowance)

**Quota exhaustion**:
A refusal that waiting will not fix within the run, such as a spent daily allowance or billing limit.
_Avoid_: rate limit (that one clears in seconds)

**Response cache**:
Stored responses keyed by the exact request, replayed instead of calling the provider.
_Avoid_: memo, semantic cache

## Scoring

**Metric**:
A per-case score with a direction (higher or lower is better) and a scale (bounded in 0 to 1, or unbounded).
_Avoid_: KPI, measure

**Aggregate**:
A metric's mean over the cases where it is defined, reported with that number of cases and a confidence interval.
_Avoid_: summary statistic, score (alone)

**Groundedness**:
The share of an answer's claims supported by a cited document the system actually retrieved.
_Avoid_: faithfulness, hallucination rate

**Abstention**:
An answer that declines because the corpus does not contain the answer.
_Avoid_: refusal (refusal also covers policy refusals)

**Failure category**:
The single earliest pipeline stage blamed for a failed case, from a fixed ordered list.
_Avoid_: error type, bug class

**Judge**:
Anything that scores an answer beyond string checks, deterministic or model-based, recorded with its own identity and version.
_Avoid_: grader, evaluator (the whole engine is the evaluator)

**Rubric judge**:
A judge that answers fixed yes or no criteria (correct, supported) about one answer.
_Avoid_: pointwise grader

**Pairwise judge**:
A judge that picks the better of two answers to the same case, asked in both orders.
_Avoid_: preference model, arena

**Position consistency**:
The share of pairwise comparisons whose winner stays the same when the two answers swap places.
_Avoid_: order robustness

**Calibration set**:
Answers with reference labels, used to measure how well a judge agrees with them.
_Avoid_: gold set, golden labels

## Agents

**Trajectory**:
The ordered tool calls an agent made while working on one case.
_Avoid_: rollout, episode, path

**Reference trajectory**:
The tool sequence an expert would use for a case, written from the task, not from a recorded run.
_Avoid_: golden path, expected trace

**Tool schema**:
The JSON Schema that a tool's arguments must satisfy.
_Avoid_: signature, spec

**Step efficiency**:
How close an agent's number of steps is to the reference's, capped at 1.
_Avoid_: speed, cost

## Comparing runs

**Baseline**:
The run a candidate is compared against, usually the committed results of the main branch.
_Avoid_: control, reference run

**Candidate**:
The run being judged for regressions.
_Avoid_: treatment, new run

**Tolerance**:
The smallest change in a metric that matters, absolute for bounded metrics and relative for unbounded ones.
_Avoid_: threshold (in prose), margin

**Paired test**:
A significance test on the per-case differences between baseline and candidate.
_Avoid_: A/B test

**Verdict**:
The outcome for one metric: improved, regressed, neutral, inconclusive, not comparable, or new.
_Avoid_: status, result

**Gate**:
The check that fails a build when any gated metric's verdict is regressed.
_Avoid_: quality bar, guard

**Attribution**:
The list of cases that individually moved a regressed metric.
_Avoid_: blame list, root cause
