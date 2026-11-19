# Responses are cached by a hash of the exact request; hits are not latency

Status: accepted (Step 2)

Reruns of an unchanged suite should cost nothing and replay the same answers,
including for nondeterministic models, so a rescore or a report rebuild never
calls the provider. The cache key is the SHA-256 of canonical JSON holding the
provider, model, messages and every output-affecting parameter (temperature,
output cap, seed, response schema) plus a format version; transport settings
such as timeouts are excluded. Any change to the prompt or parameters is a miss
by construction, so the cache can never serve an answer to a different request.
A hit is marked on its span and excluded from latency aggregates, while its
stored usage is kept so cost reflects the response that was produced. The
alternative, a semantic or prompt-template cache, would trade exactness for hit
rate, which is the wrong trade for an evaluator.
