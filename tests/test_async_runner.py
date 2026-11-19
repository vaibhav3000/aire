"""AsyncExperimentRunner: ordering, bounded concurrency, real cancellation, span trees."""

import asyncio
import json
from pathlib import Path
from typing import Any

from aire.llm.base import SUTResult
from aire.llm.deterministic import ExtractiveRAG
from aire.runner import ExperimentRunner, RunnerConfig
from aire.runner.async_runner import AsyncExperimentRunner, AsyncRunnerConfig, AsyncSystemUnderTest
from aire.schema import Span
from aire.tracing.storage import load_run_traces


def _dataset(tmp_path: Path, n: int) -> Path:
    path = tmp_path / "cases.jsonl"
    rows = [
        {"case_id": f"c{i:02d}", "input": f"question {i}", "expected_keywords": ["answer"]}
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


class SlowEcho(AsyncSystemUnderTest):
    """Answers after a per-case delay and tracks how many calls overlap."""

    name = "slow_echo"

    def __init__(self, delays: dict[str, float] | None = None, hang_on: str | None = None) -> None:
        self.config = {"kind": "test"}
        self.delays = delays or {}
        self.hang_on = hang_on
        self.in_flight = 0
        self.max_in_flight = 0
        self.cancelled: list[str] = []

    async def ainvoke(
        self, case_input: str, case_context: dict[str, Any] | None = None
    ) -> SUTResult:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if case_input == self.hang_on:
                await asyncio.sleep(30)
            await asyncio.sleep(self.delays.get(case_input, 0.01))
        except asyncio.CancelledError:
            self.cancelled.append(case_input)
            raise
        finally:
            self.in_flight -= 1
        child = Span(kind="retrieval", name="lookup", started_at=0.0, latency_ms=1.0)
        return SUTResult(answer=f"answer to {case_input}", spans=[child])


def test_traces_are_stored_in_dataset_order_whatever_finishes_first(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, 6)
    # Earlier cases are slower, so they finish last.
    sut = SlowEcho({f"question {i}": 0.06 - 0.01 * i for i in range(6)})
    outcome = AsyncExperimentRunner(sut, AsyncRunnerConfig(concurrency=6)).run(
        dataset, runs_root=tmp_path / "runs"
    )
    traces = load_run_traces(outcome.run_dir)
    assert [t.case_id for t in traces] == [f"c{i:02d}" for i in range(6)]
    assert outcome.n_cases == 6
    assert outcome.n_failed == 0
    assert outcome.cases_per_s > 0
    assert (outcome.run_dir / "manifest.json").is_file()


def test_concurrency_is_bounded(tmp_path: Path) -> None:
    sut = SlowEcho({f"question {i}": 0.03 for i in range(12)})
    AsyncExperimentRunner(sut, AsyncRunnerConfig(concurrency=3)).run(
        _dataset(tmp_path, 12), runs_root=tmp_path / "runs"
    )
    assert sut.max_in_flight == 3


def test_a_timeout_cancels_the_async_system(tmp_path: Path) -> None:
    sut = SlowEcho(hang_on="question 1")
    outcome = AsyncExperimentRunner(sut, AsyncRunnerConfig(concurrency=2, timeout_s=0.2)).run(
        _dataset(tmp_path, 3), runs_root=tmp_path / "runs"
    )
    traces = {t.case_id: t for t in load_run_traces(outcome.run_dir)}
    assert sut.cancelled == ["question 1"]
    assert traces["c01"].error is not None and "timeout" in traces["c01"].error
    assert outcome.n_failed == 1
    assert traces["c00"].error is None


def test_child_spans_hang_under_the_attempt_span(tmp_path: Path) -> None:
    outcome = AsyncExperimentRunner(SlowEcho(), AsyncRunnerConfig()).run(
        _dataset(tmp_path, 1), runs_root=tmp_path / "runs"
    )
    (trace,) = load_run_traces(outcome.run_dir)
    root, child = trace.spans
    assert root.parent_id is None and root.span_id
    assert child.parent_id == root.span_id
    assert child.kind == "retrieval"


def test_sync_systems_give_the_same_answers_as_the_sync_runner(tmp_path: Path) -> None:
    corpus = {"doc_a": "Refunds are issued within 30 days. Plans renew monthly."}
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"case_id": "r", "input": "When are refunds issued?"}) + "\n", encoding="utf-8"
    )
    sut = ExtractiveRAG("v", corpus, top_k=1)
    sync_dir = ExperimentRunner(sut, RunnerConfig()).run(path, runs_root=tmp_path / "sync")
    outcome = AsyncExperimentRunner(sut, AsyncRunnerConfig()).run(
        path, runs_root=tmp_path / "async"
    )
    assert [t.final_answer for t in load_run_traces(sync_dir)] == [
        t.final_answer for t in load_run_traces(outcome.run_dir)
    ]
