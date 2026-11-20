# Zero-dependency core; SDKs are optional extras faked at the adapter seam

Status: accepted (Step 2)

`pip install aire` must keep working with no third-party packages, because the
gate runs in CI images and other teams' pipelines where every dependency is a
supply-chain and upgrade cost. The core therefore includes a small asyncio
HTTP/1.1 client and an OpenAI-compatible adapter on top of it (enough for
Ollama, vLLM and any OpenAI-style endpoint), while the Gemini, OpenAI and
Anthropic adapters and the OpenTelemetry SDK bridge live behind the extras
`aire[gemini]`, `aire[openai]`, `aire[anthropic]` and `aire[otel]`, imported
only when such an adapter is constructed. Tests fake the SDK clients at the
adapter seam rather than at the HTTP layer: the 2026 SDKs moved to httpx2,
which respx and pytest-httpx do not intercept, and faking our own seam keeps
the tests independent of SDK transport internals. A contract test checks the
fakes against the real SDK classes, and an import test proves the core loads no
SDK module.
