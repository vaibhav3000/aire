"""Experiment runner: what goes into a span's latency, failed attempts, token accounting."""

import json
import time
from pathlib import Path
from typing import Any

from aire.llm.base import SUTResult, SystemUnderTest
from aire.llm.openai_compat import OpenAIChatResponder
from aire.llm.provider import LLMResponse, OpenAICompatProvider, ProviderError
from aire.runner.runner import ExperimentRunner, RunnerConfig
from aire.schema import Trace
from aire.tracing.storage import load_run_traces

CORPUS = {"doc_a": "The refund window is 30 days for annual plans."}


def _dataset(tmp_path: Path, n: int = 1) -> Path:
    ds = tmp_path / "ds.jsonl"
    rows = [{"case_id": f"c{i}", "input": "refund window?", "must_cite": False} for i in range(n)]
    ds.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return ds


def _run(sut: SystemUnderTest, tmp_path: Path, n: int = 1, **config: Any) -> list[Trace]:
    runner = ExperimentRunner(sut, RunnerConfig(**{"timeout_s": 5.0, **config}))
    return load_run_traces(runner.run(_dataset(tmp_path, n), runs_root=tmp_path / "runs"))


class _PacedSUT(SystemUnderTest):
    """Waits for a rate-limit slot in before_invoke, then answers at once."""

    name = "paced"

    def __init__(self) -> None:
        self.paced = 0

    def before_invoke(self) -> None:
        self.paced += 1
        time.sleep(0.4)

    def invoke(self, case_input: str, case_context: dict[str, Any] | None = None) -> SUTResult:
        return SUTResult(answer="I don't know.")


def test_client_side_pacing_runs_before_the_timer(tmp_path: Path) -> None:
    sut = _PacedSUT()
    traces = _run(sut, tmp_path)
    assert sut.paced == 1
    assert traces[0].spans[0].latency_ms < 200


class _BackoffSUT(SystemUnderTest):
    """Spends 350 ms inside invoke, 300 ms of it in a reported rate-limit wait."""

    name = "backoff"

    def invoke(self, case_input: str, case_context: dict[str, Any] | None = None) -> SUTResult:
        time.sleep(0.35)
        return SUTResult(answer="I don't know.", wait_ms=300.0, span_attributes={"attempts": 2})


def test_reported_backoff_is_recorded_and_excluded_from_latency(tmp_path: Path) -> None:
    span = _run(_BackoffSUT(), tmp_path)[0].spans[0]
    assert span.latency_ms < 200
    assert span.attributes["wait_ms"] == 300.0
    assert span.attributes["attempts"] == 2


class _SlowSUT(SystemUnderTest):
    name = "slow"

    def invoke(self, case_input: str, case_context: dict[str, Any] | None = None) -> SUTResult:
        time.sleep(1.0)
        return SUTResult(answer="late")


def test_a_timed_out_attempt_still_gets_a_span(tmp_path: Path) -> None:
    trace = _run(_SlowSUT(), tmp_path, timeout_s=0.2)[0]
    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span.error is not None and "timeout" in span.error.lower()
    assert span.latency_ms >= 150
    assert trace.usage.token_source == "none"
    assert trace.usage.total_tokens == 0


class _ErrorResultSUT(SystemUnderTest):
    name = "error_result"

    def invoke(self, case_input: str, case_context: dict[str, Any] | None = None) -> SUTResult:
        return SUTResult(answer=None, prompt_text="a long prompt " * 40, error="HTTP 429")


def test_a_failed_call_gets_no_estimated_tokens(tmp_path: Path) -> None:
    trace = _run(_ErrorResultSUT(), tmp_path)[0]
    assert trace.error == "HTTP 429"
    assert trace.usage.token_source == "none"
    assert trace.usage.total_tokens == 0


class _FakeProvider(OpenAICompatProvider):
    """Real provider class with chat() replaced, so no request leaves the process."""

    def __init__(self, error: ProviderError | None = None) -> None:
        super().__init__(base_url="http://127.0.0.1:9", api_key_env="UNUSED", model="fake-model")
        self.error = error

    def chat(self, system: str, user: str, retries: int | None = None) -> LLMResponse:
        if self.error is not None:
            raise self.error
        return LLMResponse(
            content="Refunds take 30 days [doc:doc_a].",
            prompt_tokens=100,
            completion_tokens=10,
            model="fake-model",
            finish_reason="stop",
            token_source="api",
            attempts=2,
            backoff_s=1.5,
        )


def test_llm_responder_paces_outside_the_timed_call(tmp_path: Path) -> None:
    sut = OpenAIChatResponder("llm", CORPUS, _FakeProvider(), min_call_interval_s=0.3)
    started = time.perf_counter()
    traces = _run(sut, tmp_path, n=2)
    assert time.perf_counter() - started >= 0.3
    assert all(trace.spans[0].latency_ms < 150 for trace in traces)


def test_llm_responder_reports_provider_attempts_and_backoff(tmp_path: Path) -> None:
    sut = OpenAIChatResponder("llm", CORPUS, _FakeProvider())
    result = sut.invoke("How long is the refund window?")
    assert result.wait_ms == 1500.0
    assert result.span_attributes["provider_attempts"] == 2
    assert result.usage is not None and result.usage["token_source"] == "api"


def test_llm_responder_keeps_backoff_of_a_failed_call(tmp_path: Path) -> None:
    error = ProviderError("HTTP 503: unavailable", status=503, attempts=4, backoff_s=7.0)
    sut = OpenAIChatResponder("llm", CORPUS, _FakeProvider(error=error))
    result = sut.invoke("How long is the refund window?")
    assert result.error is not None and "HTTP 503" in result.error
    assert result.wait_ms == 7000.0
    assert result.span_attributes["provider_attempts"] == 4
