# The model client owns retries and rate limiting; SDK retries are off

Status: accepted (Step 2)

Every provider SDK ships its own retry loop (OpenAI and Anthropic retry twice
by default with their own backoff, google-genai 3.x makes one attempt unless
`retry_options` is set), and none of them shares a rate budget across
concurrent calls or reports how long it waited. AIRE therefore constructs every
SDK client with retries disabled (`max_retries=0`, no `retry_options`) and does
all retrying in `aire.providers.client.ModelClient`, which also owns the shared
adaptive token bucket, the quota circuit and the response cache. This keeps one
retry policy for every provider, keeps waits out of measured latency, and makes
every attempt visible as its own span. The cost is that we maintain the error
classification per provider ourselves, which the adapter contract tests cover.

## Considered options

- Keep SDK retries and add a limiter on top: two stacked retry loops multiply
  attempts on a 429 storm and hide waits inside "latency".
- Use a third-party resilience library (tenacity, aiolimiter): would break the
  zero-dependency core for logic that is about a hundred lines.
