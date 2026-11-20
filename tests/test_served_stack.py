"""The whole live path, offline: async runner, LLM RAG system, model client,
stdlib HTTP adapter and real sockets, against a 127.0.0.1 server that enforces
its own rate limit, injects errors and can run out of quota.
"""

import asyncio
import json
import random
from pathlib import Path

import pytest

from aire.evaluator.engine import EvaluationEngine
from aire.providers.cache import ResponseCache
from aire.providers.client import ModelClient, RetryPolicy
from aire.providers.local import LocalOpenAICompatModel
from aire.providers.ratelimit import AdaptiveRateLimiter
from aire.runner.async_runner import AsyncExperimentRunner, AsyncRunnerConfig, RunOutcome
from aire.runner.dataset import load_eval_dataset
from aire.systems.rag import LLMRagSystem
from aire.testing.fake_server import FakeLLMServer, RetryHint
from aire.tracing.storage import load_run_traces

ROOT = Path(__file__).resolve().parents[1]
CORPUS = {
    p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "demo" / "corpus").glob("*.md"))
}


def _dataset(tmp_path: Path, n: int) -> Path:
    lines = (ROOT / "demo" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()[:n]
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def _run(
    server: FakeLLMServer,
    dataset: Path,
    runs_root: Path,
    *,
    rpm: float | None = None,
    cache: ResponseCache | None = None,
    concurrency: int = 8,
) -> tuple[RunOutcome, ModelClient]:
    client = ModelClient(
        LocalOpenAICompatModel(server.base_url, provider="openai", timeout_s=10),
        limiter=AdaptiveRateLimiter(requests_per_minute=rpm, tokens_per_minute=None),
        cache=cache,
        retry=RetryPolicy(max_attempts=6, base_delay_s=0.05, max_delay_s=5.0),
        rng=random.Random(0),
    )
    sut = LLMRagSystem("served", CORPUS, client, model="fake-extractive", deterministic=True)
    runner = AsyncExperimentRunner(sut, AsyncRunnerConfig(concurrency=concurrency, timeout_s=30))
    outcome = await runner.arun(dataset, runs_root=runs_root)
    return outcome, client


@pytest.mark.parametrize("hint", ["header", "gemini"])
def test_a_rate_limited_server_finishes_the_suite_with_zero_failures(
    tmp_path: Path, hint: RetryHint
) -> None:
    dataset = _dataset(tmp_path, 26)

    async def main() -> tuple[RunOutcome, ModelClient, FakeLLMServer]:
        # The client is told 1200 RPM, the server allows 600: it has to adapt.
        async with FakeLLMServer(
            requests_per_minute=600, service_time_s=0.01, retry_hint=hint
        ) as server:
            outcome, client = await _run(server, dataset, tmp_path / "runs", rpm=1200)
            return outcome, client, server

    outcome, client, server = asyncio.run(main())
    assert outcome.n_failed == 0
    assert server.rate_limited > 0  # the limit was really hit and absorbed
    assert server.served == 26
    assert client.stats()["rate_limited"] == server.rate_limited
    traces = load_run_traces(outcome.run_dir)
    assert all(t.final_answer for t in traces)
    assert all(t.usage.token_source == "api" for t in traces)
    results = EvaluationEngine(corpus=CORPUS).evaluate_run(traces, load_eval_dataset(dataset))
    assert results["aggregates"]["mean_errors"] == 0.0


def test_injected_server_errors_are_retried(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 12)

    async def main() -> tuple[RunOutcome, FakeLLMServer]:
        async with FakeLLMServer(fail_every=3) as server:
            outcome, _ = await _run(server, dataset, tmp_path / "runs")
            return outcome, server

    outcome, server = asyncio.run(main())
    assert outcome.n_failed == 0
    assert server.injected_failures > 0
    traces = load_run_traces(outcome.run_dir)
    retried = [
        t for t in traces if len([s for s in t.spans if s.kind == "generation" and s.parent_id]) > 1
    ]
    assert retried  # one child span per model attempt, failed ones included


def test_quota_exhaustion_fails_the_rest_without_calling_the_server(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 20)

    async def main() -> tuple[RunOutcome, FakeLLMServer]:
        async with FakeLLMServer(quota_after=5) as server:
            outcome, _ = await _run(server, dataset, tmp_path / "runs", concurrency=1)
            return outcome, server

    outcome, server = asyncio.run(main())
    assert server.served == 5
    assert server.requests == 6  # five answers and the one refusal; nothing after it
    assert outcome.n_failed == 15
    errors = [t.error for t in load_run_traces(outcome.run_dir) if t.error]
    assert all("quota" in e for e in errors)


def test_a_warm_cache_rerun_makes_no_requests(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 10)
    cache = ResponseCache(tmp_path / "cache")

    async def main() -> tuple[int, int, RunOutcome]:
        async with FakeLLMServer() as server:
            await _run(server, dataset, tmp_path / "cold", cache=cache)
            cold = server.requests
            outcome, _ = await _run(server, dataset, tmp_path / "warm", cache=cache)
            return cold, server.requests, outcome

    cold, after_warm, warm = asyncio.run(main())
    assert cold == 10
    assert after_warm == cold
    traces = load_run_traces(warm.run_dir)
    assert all(t.spans[0].attributes.get("cache_hit") is True for t in traces)
    rows = EvaluationEngine(corpus=CORPUS).evaluate_run(traces, load_eval_dataset(dataset))
    assert rows["aggregates"]["n_cache_hits"] == 10
    assert "mean_latency_ms" not in rows["aggregates"]


def test_the_served_answers_match_the_extractive_demo_system(tmp_path: Path) -> None:
    """The fake server answers like ExtractiveRAG would from the same top document."""
    dataset = _dataset(tmp_path, 3)

    async def main() -> RunOutcome:
        async with FakeLLMServer() as server:
            outcome, _ = await _run(server, dataset, tmp_path / "runs")
            return outcome

    traces = load_run_traces(asyncio.run(main()).run_dir)
    first = json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
    assert traces[0].case_id == first["case_id"]
    assert "[doc:doc_refunds]" in (traces[0].final_answer or "")
