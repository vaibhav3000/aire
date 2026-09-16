"""Experiment runner: drives a SystemUnderTest over an evaluation set.

Responsibilities:
- invoke the SUT once per case, wrap the result into a validated Trace
- record spans (retrieval/generation) with real latencies
- retry policy (attempts, backoff) for transient invocation errors
- timeout enforcement (each invocation runs in a watchdog thread)
- token usage accounting (heuristic unless the SUT provides real counts)
- persist manifest + traces via RunStore

The runner never imports the SUT's dependencies: the SUT is injected, so the
same runner evaluates a deterministic demo system or a real LLM app.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..llm.base import SystemUnderTest
from ..schema import Span, Timer, Trace, Usage, estimate_tokens, new_id
from .dataset import EvalCase, load_eval_dataset
from ..tracing.storage import RunStore


class InvocationTimeout(Exception):
    pass


def _invoke_with_timeout(sut: SystemUnderTest, case_input: str, timeout_s: float) -> Any:
    """Run sut.invoke in a worker thread; raise InvocationTimeout on overrun.

    Threads cannot be killed in Python, so a hung call leaks its thread —
    documented honestly; the watchdog only bounds the runner's wait.
    """
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["value"] = sut.invoke(case_input)
        except Exception as exc:  # noqa: BLE001 - recorded as trace error
            result["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise InvocationTimeout(f"invocation exceeded {timeout_s:.0f}s timeout")
    if "error" in result:
        raise RuntimeError(result["error"])
    if "value" not in result:
        raise RuntimeError("invocation produced no result")
    return result["value"]


@dataclass
class RunnerConfig:
    timeout_s: float = 30.0
    max_retries: int = 0
    retry_backoff_s: float = 0.2
    heuristic_tokens: bool = True  # estimate tokens when the SUT does not report usage


class ExperimentRunner:
    """Runs one experiment: (system version) x (dataset) -> stored run."""

    def __init__(self, sut: SystemUnderTest, config: RunnerConfig | None = None) -> None:
        self.sut = sut
        self.config = config or RunnerConfig()

    def run(
        self,
        dataset_path: str | Path,
        runs_root: str | Path = "runs",
        run_name: str | None = None,
    ) -> Path:
        cases: list[EvalCase] = load_eval_dataset(dataset_path)
        run_id = new_id("run")
        name = run_name or f"{self.sut.name}_{run_id}"
        store = RunStore(runs_root, run_id, name)
        store.write_manifest(
            created_at=time.time(),
            system_config=self.sut.config,
            dataset_path=dataset_path,
            n_cases=len(cases),
        )

        for case in cases:
            trace = self._run_case(case, run_id)
            store.append_trace(trace)
        store.flush_traces()
        return store.run_dir

    def _run_case(self, case: EvalCase, run_id: str) -> Trace:
        trace = Trace(
            run_id=run_id,
            case_id=case.case_id,
            system_config=self.sut.config,
            started_at=time.time(),
        )
        attempts = self.config.max_retries + 1
        last_error: str | None = None
        for attempt in range(attempts):
            try:
                with Timer() as gen_timer:
                    result = _invoke_with_timeout(self.sut, case.input, self.config.timeout_s)
                trace.final_answer = result.answer
                trace.retrieved_doc_ids = list(result.retrieved_doc_ids)
                trace.tool_calls = list(result.tool_calls)
                if result.error:
                    # The system under test failed internally (e.g. provider
                    # error); surface it as the trace error so the failure
                    # taxonomy classifies it as invocation_error, not as an
                    # empty answer.
                    trace.error = result.error
                trace.spans.append(
                    Span(
                        kind="generation",
                        name=f"{self.sut.name}.invoke",
                        started_at=trace.started_at,
                        latency_ms=gen_timer.latency_ms,
                        attributes={"attempt": attempt + 1},
                    )
                )
                if result.usage:
                    # Real API usage reported by the system under test.
                    trace.usage = Usage(
                        prompt_tokens=int(result.usage.get("prompt_tokens", 0)),
                        completion_tokens=int(result.usage.get("completion_tokens", 0)),
                        token_source=result.usage.get("token_source", "api"),
                    )
                else:
                    prompt_tokens = (
                        estimate_tokens(result.prompt_text)
                        if self.config.heuristic_tokens
                        else 0
                    )
                    completion_tokens = (
                        estimate_tokens(result.answer or "") if self.config.heuristic_tokens else 0
                    )
                    trace.usage = Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        token_source="heuristic" if self.config.heuristic_tokens else "none",
                    )
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - recorded, retried, classified
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts - 1:
                    time.sleep(self.config.retry_backoff_s * (attempt + 1))

        if last_error is not None:
            trace.error = last_error
        return trace
