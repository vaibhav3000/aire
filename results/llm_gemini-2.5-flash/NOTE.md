# Live run, cut short by the provider's daily quota

- **When:** 2026-11-19 21:05 UTC (`manifest.json` `created_at`).
- **What:** `python scripts/run_llm_eval.py --provider gemini --model gemini-2.5-flash --rpm 10 --concurrency 4`,
  through the google-genai 3.6.1 adapter on the `v1beta` API, temperature 0, at most 256 output tokens,
  local idf retrieval (top 3) over the 150-case suite.
- **What happened:** 20 cases were answered. The next two calls returned HTTP 429 "You exceeded your
  current quota", which aire classified as an exhausted quota (for a Gemini body that means a per-day
  `quotaId`). The quota circuit then failed the other 128 cases without sending them. In total the
  provider received 22 generation requests. An earlier attempt the same evening sent 2 requests that the
  `v1` API rejected (system instructions need `v1beta`), which led to commit 64f04c8.
- **Tokens (API counts, answered cases):** 12,369 prompt and 3,174 completion, of which 2,397 were
  reasoning tokens.
- **The answered cases** are the first 20 lines of `demo/eval_set.jsonl` in dataset order (concurrency
  4): 16 succeeded, 2 were incomplete, 1 ungrounded and 1 a retrieval miss; all 3 unanswerable questions
  among them were declined correctly. This is 20 of 150 cases and not a sample of the suite, so it
  supports no comparison with the deterministic systems.

`traces.jsonl` and `manifest.json` are the replayable record; `eval_results.json` scores all 150 cases
with the 130 unanswered ones as invocation errors. The top-level `results/llm_*.json` files and
`reports/llm_vs_deterministic_v2.html` still hold the historical 26-case run and were deliberately not
replaced. A rerun with the same command after the quota resets reuses the 20 cached answers from
`runs/llm_cache` (local, not committed) and sends only the remaining requests.
