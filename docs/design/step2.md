# AIRE Step 2 design: production evaluation at scale

Status: accepted. Builds on the Step 1 branch (`upgrade/nov-2026`), where every
audit finding was fixed and the suite stood at 152 tests. Terms in **bold** are
defined in [docs/GLOSSARY.md](../GLOSSARY.md); the decisions that are hard to
reverse are recorded as ADRs in [docs/adr/](../adr/).

## 1. Why this step exists

After Step 1, AIRE scores recorded traces correctly, but four things keep it
from being an evaluation system a team would run against real models:

1. **Throughput and cost control.** The runner calls one case at a time in a
   worker thread. A 150 case suite against a rate-limited API takes minutes,
   a timed-out thread keeps spending quota, and every rerun pays again.
2. **One provider path.** Only an OpenAI-compatible stdlib client exists.
   Gemini, OpenAI and Anthropic each have their own SDK, error model, usage
   fields and structured-output API.
3. **Point estimates on 26 cases.** The gate compares two means against a
   tolerance. With 26 cases a single flipped case moves a bounded mean by
   almost four points, so the gate cannot tell a regression from noise.
4. **Answers only.** Agents are judged on their final answer; the path they
   took (which tools, which arguments, how many steps) is not scored.

The step adds five capabilities. Each section below states the approaches
that were considered, the decision, and how it is proved offline.

## 2. Shape of the system after Step 2

```
                    eval set (150 cases)          agent cases + references
                            |                               |
   SystemUnderTest ---------+                               |
   (sync or async)          v                               v
        |          AsyncExperimentRunner            trajectory ingestion
        |          (bounded concurrency)        (repo-engineer JSON/JSONL,
        v                   |                    OTLP JSON spans)
   ModelClient  <-----------+                               |
   cache, rate limiter,     |                               |
   retries, usage           v                               v
        |               RunStore: manifest.json, traces.jsonl (canonical)
        v                            |
   ChatModel adapters                +--> OTel export (OTLP, JSON lines)
   Gemini | OpenAI | Anthropic |     v
   local OpenAI-compatible    EvaluationEngine: deterministic metrics,
   (stdlib) | scripted fake   trajectory metrics, bootstrap CIs
                                     |            \
                                     |             judges (rubric, pairwise)
                                     v             calibrated by kappa
                               compare_runs: paired tests per metric
                                     |
                      aire-gate (exit code), HTML diff report,
                      PR comment (fork-safe GitHub workflows)
```

The trace stays the single source of truth: everything downstream reads
`traces.jsonl`, so a run is recorded once and can be rescored, compared,
exported and reported any number of times (ADR 0003).

### Module map (new modules in italics)

| Package | Modules | Role |
| --- | --- | --- |
| `aire.providers` | *types, base, errors, http, local, gemini, openai, anthropic, fake, ratelimit, cache, client* | Talking to models: one request and response type, one adapter per provider, and the **model client** that owns pacing, retries, caching and usage. |
| `aire.runner` | runner, dataset, *async_runner* | Driving a system under test over an eval set. |
| `aire.systems` | *rag* | The LLM-backed RAG system (async) used for live and served runs. |
| `aire.evaluator` | engine, metrics, judges, *stats, trajectory, calibration* | Scoring. |
| `aire.agents` | *ingest* | Turning agent logs into aire traces. |
| `aire.telemetry` | *genai, otlp_json, otel_sdk* | OpenTelemetry export and ingestion. |
| `aire.regression` | compare, gate | Paired comparison and the CI gate. |
| `aire.report` | html, *markdown* | HTML diff report and the PR comment body. |
| `aire.ci` | *github* | Sticky PR comment upsert through the REST API. |
| `aire.testing` | *fake_server* | A 127.0.0.1 OpenAI-compatible server that enforces rate limits. |
| `aire` | *jsonschema, cli* | A small JSON Schema validator and the `aire` command. |

The core (`pip install aire`) still has no runtime dependency. Each SDK is an
optional extra: `aire[gemini]`, `aire[openai]`, `aire[anthropic]`,
`aire[otel]`, and `aire[all]` for all of them (ADR 0002).

## 3. Async runner, rate limiting, retries and the response cache

### Problem

A provider enforces requests per minute (RPM) and tokens per minute (TPM).
Going faster than the limit earns HTTP 429s; going one request at a time
wastes most of the budget. Retries must not stampede, must honour what the
server asks for, and must stop at once when the daily quota is gone. Reruns
of an unchanged suite should cost nothing.

### Approaches considered

- *Thread pool around the existing sync client.* Small change, but a timed-out
  thread cannot be cancelled and keeps the request alive (Step 1 follow-up 8),
  and per-thread rate limiting is hard to share.
- *Rely on the SDKs' built-in retries.* Every SDK retries differently (OpenAI
  and Anthropic retry twice by default, google-genai 3.x not at all unless
  configured), none of them shares a budget across concurrent calls, and
  their waits would be invisible to latency accounting.
- *asyncio with one shared client that owns pacing and retries.* Chosen.

### Decision

- **`AsyncExperimentRunner`** runs cases with bounded concurrency (a fixed
  pool of worker tasks reading a queue, default 8). Async systems run on the
  event loop and are cancelled for real on timeout (`asyncio.timeout`); sync
  systems run through `asyncio.to_thread`, keeping their old semantics.
  Traces are appended as each case finishes (crash safety) and the file is
  rewritten in dataset order at the end, so the output does not depend on
  completion order. It returns a `RunOutcome` with wall time, throughput and
  failure counts.
- **`ModelClient`** wraps one `ChatModel` adapter and is shared by every
  concurrent case. For each request it: looks up the **response cache**,
  waits for the **rate limiter**, makes one attempt, classifies a failure,
  and retries with backoff. It returns the response together with one record
  per attempt (start, latency, error kind, wait before it), which the system
  turns into child spans of the case's span.
- **Adaptive token bucket.** Two buckets, requests and tokens, refilled
  continuously from the configured RPM and TPM. A request takes one request
  token and its estimated token cost (prompt characters / 4 plus the output
  cap); after the call the estimate is reconciled with the real usage, so the
  token bucket tracks what the provider actually counted. On a 429 the
  allowed rate is cut in half (never below a floor) and all callers pause
  until the server's retry time has passed; each success raises the rate
  again by a small additive step, back up to the configured ceiling. This is
  the additive-increase, multiplicative-decrease rule TCP uses for the same
  problem. Waiters are served first come, first served.
- **Retries.** Retried: 408, 429, 500, 502, 503, 504, timeouts, connection
  errors. The wait is the server's own request when it gives one
  (`Retry-After` seconds or HTTP date, Gemini `retryDelay`, OpenAI
  `retry-after-ms`), otherwise exponential backoff with full jitter: a uniform
  draw from zero to `min(cap, base * 2**attempt)`. Every retry goes back
  through the rate limiter, so a wave of 429s cannot turn into a wave of
  simultaneous retries.
- **Fail fast.** A 429 that reports an exhausted quota (OpenAI
  `insufficient_quota`, a Gemini per-day `quotaId`, Anthropic billing errors)
  opens a circuit: that call and every later call through the same client
  fail at once without touching the network. A server wait longer than the
  cap also fails the call instead of sleeping past the budget. Client errors
  (400, 401, 403, 404, 422) are never retried.
- **Response cache.** The key is the SHA-256 of a canonical JSON document
  holding the provider, model, messages and every parameter that changes the
  output (temperature, output cap, seed, response schema), plus a cache format
  version. Timeouts and retry settings are not part of the key. Entries are
  JSON files written atomically under a two-character fan-out directory.
  Identical requests in flight at the same time share one call (single
  flight). A cache hit is recorded on the span (`cache_hit`); the engine
  leaves cache hits out of latency, since a lookup says nothing about the
  model, and the stored usage stays attached so cost reflects what the
  response consumed when it was produced (ADR 0006).

### Offline proof

`aire.testing.fake_server` is an asyncio HTTP server on 127.0.0.1 that speaks
the OpenAI chat-completions protocol, answers RAG prompts extractively (so
answers are deterministic and scorable), simulates service time, and enforces
its own RPM and TPM with 429 responses carrying `Retry-After`. It can also
inject 503s and quota-exhausted 429s on a schedule. Tests drive the whole
stack (async runner, LLM RAG system, model client, stdlib HTTP adapter,
sockets) against it and require zero failed cases. `demo/bench_runner.py`
records sequential against concurrent throughput, the number of 429s absorbed,
and a warm-cache rerun that makes no requests at all, in
`results/runner_benchmark.json` with the machine and configuration. Timing
numbers come only from that script's real runs.

## 4. Provider adapters

### Decision

One protocol, `ChatModel`: a `provider` name (the OpenTelemetry
`gen_ai.provider.name` value), an optional server address, and
`async complete(request) -> ChatResponse`, which makes exactly one attempt and
raises `ProviderError` with a `kind` (rate_limited, quota_exhausted,
transient, timeout, connection, client, malformed) and any server-requested
wait. Retrying is the model client's job, so every SDK adapter turns the
SDK's own retries off (ADR 0001).

| Adapter | Transport | Structured output | Usage mapping |
| --- | --- | --- | --- |
| `LocalOpenAICompatModel` (Ollama, vLLM, any OpenAI-style server) | stdlib asyncio HTTP/1.1 client, http and https | `response_format` json_schema | `prompt_tokens`, `completion_tokens`, `cached_tokens`, `reasoning_tokens` |
| `OpenAIModel` (`openai` 3.x) | `AsyncOpenAI(max_retries=0)` | `response_format` json_schema, strict | same as above |
| `AnthropicModel` (`anthropic` 2.x) | `AsyncAnthropic(max_retries=0)` | `output_config={"format": {"type": "json_schema", ...}}` | input = `input_tokens` + cache reads + cache writes; output = `output_tokens` |
| `GeminiModel` (`google-genai` 3.x) | `client.aio`, no `retry_options` (one attempt) | `response_mime_type` json plus `response_json_schema` | input = `prompt_token_count`; output = `candidates_token_count` + `thoughts_token_count` |
| `ScriptedChatModel` | in memory | parsed from scripted text | scripted |

Provider facts that shape the adapters (from the November 2026 SDK notes):
Anthropic 2.x removed `temperature` from `messages.create`, so that adapter
omits it and records "provider default sampling" in the run config; google-genai
3.x defaults to the `v1` API and accepts an `api_version` override; every SDK
now uses httpx2, which HTTP-level mocking libraries do not intercept.

Structured output is validated by aire itself against the request's JSON
Schema (`aire.jsonschema`), because the SDKs disagree on failure behaviour
(Gemini returns `parsed=None` silently, Anthropic's helper raises). A response
that is not valid JSON for the schema is a `malformed` error, never retried.

### Testing

Adapters take an injected SDK client. Tests pass fakes shaped like the SDK
responses and exceptions, so the mapping code is tested without the network
and without httpx2. A contract test imports the real SDK packages (installed
by the dev extra) and checks that the attributes the adapters read exist on
the real exception and response classes. Another test imports the core in a
fresh interpreter and asserts that no SDK module was loaded.

## 5. Agentic trajectory evaluation

### Decision

A **trajectory** is the ordered list of tool calls in a trace. A case may
carry a **reference trajectory** (`expected_tool_sequence`), and the engine
may be given the **tool schemas** (JSON Schema per tool). Per case:

- `tool_selection_accuracy`: length of the longest common subsequence of the
  actual and reference tool names, divided by the reference length (did the
  agent make the reference's tool choices, in order).
- `tool_call_precision`: the same subsequence length divided by the number of
  actual calls (how much of what it did was on the reference path).
- `step_efficiency`: `min(1, reference steps / actual steps)`.
- `tool_args_schema_valid`: the share of calls whose arguments validate
  against the tool's schema; an unknown tool counts as invalid.

The existing `tool_args_valid` (the runtime's own verdict) stays. A schema
violation makes the case a `tool_error` in the failure taxonomy; a longer path
lowers efficiency but is not a failure.

Ingestion (`aire.agents.ingest`) reads repo-engineer trace files, either one
JSON object per file (its current `results/benchmark_traces/*.json`) or one
per line (JSONL), and maps actions and observations to aire tool calls, the
final state to the trace error, and LLM usage to token counts. The four real
traces are copied into `demo/agent/` with their source commit recorded.
References are written from each task's description and the runtime's fixed
analyze and verify steps, not from the recorded runs. A deterministic,
documented mutation script derives a seeded-defect variant (invalid argument
types, a repeated search, a skipped test run) so the metrics and the gate can
be shown catching agent regressions. With four tasks no paired test can reach
significance, and the comparison says so.

## 6. Hybrid scoring, statistics and the 150 case suite

### Statistics

- Every per-run mean gets a 95% percentile bootstrap interval
  (`ci95_<metric>`), seeded, so committed results are reproducible.
- Comparisons pair cases by id. Binary per-case metrics (every value 0 or 1
  in both runs) use McNemar's exact test on the discordant pairs; the rest use
  a paired bootstrap of the mean difference, whose two-sided p-value comes
  from the same resamples as its interval, so the interval excludes zero
  exactly when p is below alpha. Holm-adjusted p-values are reported for the
  family of gated metrics.
- **Verdict rule.** Direction-adjusted difference `d` (positive is better),
  tolerance `t` (absolute for bounded scores, relative to the baseline mean
  for latency, tokens and cost), significance from the paired test at alpha
  (default 0.05):
  - `regressed`: `d < -t` and significant;
  - `improved`: `d > t` and significant;
  - `inconclusive`: `|d| > t` but not significant;
  - `neutral`: `|d| <= t`.

  The gate fails on `regressed`; `--fail-on-inconclusive` also fails on an
  inconclusive drop. Fewer than 10 paired values cannot support a resampling
  test, so such rows fall back to the Step 1 tolerance rule and say so
  (`test: "point"`). Point mode stays available as `--method point`.
- Cohen's kappa measures judge agreement with labels.

### Judges

- `RubricJudge`: asks a model whether an answer is **correct** (it answers
  the question in agreement with the corpus) and **supported** (each claim is
  backed by a cited, retrieved document), with a JSON Schema output and a
  short rationale. The judge's model, prompt version and parameters are
  stored with every verdict.
- `PairwiseJudge`: compares two answers twice, once in each order. A verdict
  counts only when both orders agree; disagreement is a tie and is reported as
  a position-consistency rate.
- `LexicalRubricJudge`: the deterministic baseline with the same interface
  (correct when keyword recall is at least 0.5, supported when groundedness is
  at least 0.5 and something is cited). It shows what the lexical proxies
  miss.
- **Calibration.** `demo/judge_calibration/labels.jsonl` holds labels for a
  stratified subset of answers (cases from every topic, all three demo
  versions). `demo/calibrate_judges.py` reports kappa, raw agreement and the
  confusion matrix per criterion for each judge. The labels were drafted
  during development and are marked unreviewed until a person confirms them;
  kappa against them is reported with that status. LLM judge numbers exist
  only after an approved live run (or a local model) and are committed with
  their traces, manifest, date and model.

Judges run at evaluation time and write `judge_*` metrics into the results;
the gate treats them like any other metric. Deterministic metrics are never
replaced, only joined.

### The 150 case suite

The suite grows from 26 to 150 cases over the same twelve documents:
96 direct single-document questions (20 of them from the original suite), 18
paraphrased ones that avoid the document's vocabulary, 16 two-document
questions and 20 unanswerable ones (up from 4, so abstention is measured on
more than a handful of cases). How the
cases were written, the rules they follow and their per-topic counts are in
[docs/eval_set.md](../eval_set.md); `demo/validate_eval_set.py` enforces the
rules. The committed demo results are regenerated on the new suite, and every
changed number is reported with its source.

## 7. Observability and reporting

### OpenTelemetry (ADR 0003)

- `aire.telemetry.genai` maps a completed aire trace to spans that follow the
  GenAI semantic conventions: one `invoke_workflow` span per case, one
  `chat {model}` inference span per model attempt (`gen_ai.operation.name`,
  `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`,
  `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
  `gen_ai.usage.cache_read.input_tokens`, `error.type`), and one
  `execute_tool {name}` span per tool call. The conventions are still in
  Development status, so the attribute names are defined locally in one
  module.
- `aire.telemetry.otlp_json` encodes those spans as OTLP JSON (JSON lines,
  one `resourceSpans` document per trace) and decodes OTLP JSON back into
  aire traces, so traces from any OpenTelemetry-instrumented application can
  be evaluated. This module is pure Python.
- `aire.telemetry.otel_sdk` (extra `aire[otel]`) emits the same spans through
  the OpenTelemetry SDK with their recorded timestamps, to an OTLP endpoint, a
  JSON lines file, or both.

### HTML diff report

One self-contained file per comparison: a verdict banner, a metric table with
deltas, confidence intervals drawn as inline SVG, p-values and the test used,
failure taxonomy changes, and the cases that flipped with both answers side by
side. Light and dark themes follow the system setting, the layout works on a
phone, tables scroll inside their own container, and every value from a
results file is escaped.

### PR regression gate (ADR 0005)

- `eval-pr.yml` runs on `pull_request` with a read-only token: it evaluates the
  PR's code, gates it against the base branch's committed results, uploads
  `gate.json` and the HTML report as an artifact, and fails the check when the
  gate fails.
- `eval-pr-comment.yml` runs on `workflow_run` with `pull-requests: write`. It
  never checks out or runs PR code: it checks out the default branch, reads the
  artifact as untrusted data, renders the markdown with the trusted code
  (escaping every string), checks that the PR's head commit is the one the
  evaluation ran on, and creates or updates one comment marked with a hidden
  HTML comment. Fork PRs get the same comment safely.

## 8. Compatibility

- `ExperimentRunner`, `SystemUnderTest`, `compare_runs`, `aire-gate` and the
  results layout keep working. New fields are additive: `Span.span_id` and
  `Span.parent_id` (latency sums only root spans), `Usage.cache_read_tokens`,
  `cache_write_tokens` and `reasoning_tokens`, `SUTResult.spans`,
  `EvalCase.expected_tool_sequence`. Trace schema version 1.2; readers still
  ignore unknown keys.
- `compare_runs` gains a `stats` argument (on by default). Rows with fewer
  than ten paired values behave exactly as before, which keeps every Step 1
  fixture meaningful.
- The Step 1 sync provider (`aire.llm.provider`) stays for the sync path and
  shares its error parsing with the new adapters.

## 9. Non-goals

- Streaming responses, multi-turn conversations, and tool calling through the
  adapters (agent trajectories are ingested, not generated live).
- HTTP keep-alive pooling and proxy support in the stdlib client; use the SDK
  adapters where those matter.
- Cache eviction, a shared cache server, or distributed execution.
- Semantic groundedness with an entailment model; the lexical proxy and the
  LLM rubric judge are what this step offers.
- Live model calls in CI. CI runs offline; live runs happen only with the
  owner's approval and are committed with traces and manifest.
- Vertex AI, batch APIs, and per-request SDK option overrides.

## 10. Risks and limitations

- Bootstrap intervals treat cases as a sample from a larger population of
  questions; they say nothing about run-to-run sampling noise of a
  nondeterministic model, which needs repeated runs.
- The 150 cases were written by one author over a 12 document synthetic
  corpus; they measure this demo, not general RAG quality.
- Hand labels for judge calibration are unreviewed until a person signs them
  off, and kappa is reported with that status.
- The OpenTelemetry GenAI conventions can still change; one module holds every
  attribute name.
- The stdlib HTTP client opens a connection per request.
