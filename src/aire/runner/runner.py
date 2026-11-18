"""Experiment runner: drives a SystemUnderTest over an evaluation set.

Responsibilities:
- invoke the SUT once per case, wrap the result into a validated Trace
- record one span per attempt, failed attempts included, with real latencies;
  client-side pacing (SystemUnderTest.before_invoke) runs before the timer and
  waits the SUT reports (SUTResult.wait_ms) are subtracted from the latency
- retry policy (attempts, backoff) for transient invocation errors
- timeout enforcement (each invocation runs in a watchdog thread)
- token usage accounting (heuristic unless the SUT provides real counts;
  failed calls get token_source "none")
- persist manifest + traces via RunStore

The runner never imports the SUT's dependencies: the SUT is injected, so the
same runner evaluates a deterministic demo system or a real LLM app.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..llm.base import SUTResult, SystemUnderTest
from ..schema import Span, Timer, Trace, Usage, estimate_tokens, new_id
from ..tracing.storage import RunStore
from .dataset import EvalCase, load_eval_dataset


class InvocationTimeout(Exception):
    pass


def _invoke_with_timeout(sut: SystemUnderTest, case_input: str, timeout_s: float) -> Any:
    """Run sut.invoke in a worker thread; raise InvocationTimeout on overrun.

    Threads cannot be killed in Python, so a hung call leaks its thread
    (a documented limitation); the watchdog only bounds the runner's wait.
    """
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["value"] = sut.invoke(case_input)
        except Exception as exc:  # any SUT failure is recorded as a trace error
            result["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise InvocationTimeout(f"invocation exceeded {timeout_s:g}s timeout")
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
        *,
        corpus: Mapping[str, str] | None = None,
        path_root: str | Path | None = None,
    ) -> Path:
        """Run every case and store manifest + traces under runs_root/<run_id>.

        `corpus` (optional) is hashed into the manifest so runs over different
        documents are distinguishable. `path_root` is the directory the
        manifest's dataset_path is recorded relative to (default: the working
        directory); a dataset outside it is recorded by file name only.
        """
        cases: list[EvalCase] = load_eval_dataset(dataset_path)
        run_id = new_id("run")
        name = run_name or f"{self.sut.name}_{run_id}"
        store = RunStore(runs_root, run_id, name)
        store.write_manifest(
            created_at=time.time(),
            system_config=dict(self.sut.config),
            dataset_path=dataset_path,
            n_cases=len(cases),
            deterministic=self.sut.deterministic,
            corpus=corpus,
            path_root=path_root,
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
            system_config=dict(self.sut.config),
            started_at=time.time(),
            usage=Usage(token_source="none"),
        )
        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            self.sut.before_invoke()  # client-side pacing, deliberately untimed
            span_start = time.time()
            timer = Timer()
            try:
                with timer:
                    result = _invoke_with_timeout(self.sut, case.input, self.config.timeout_s)
            except Exception as exc:  # timeout or SUT exception: recorded, retried, classified
                error = f"{type(exc).__name__}: {exc}"
                trace.spans.append(
                    Span(
                        kind="generation",
                        name=f"{self.sut.name}.invoke",
                        started_at=span_start,
                        latency_ms=timer.latency_ms,
                        attributes={"attempt": attempt},
                        error=error,
                    )
                )
                trace.error = error
                if attempt < attempts:
                    time.sleep(self.config.retry_backoff_s * attempt)
                continue

            wait_ms = max(0.0, float(result.wait_ms))
            attributes: dict[str, Any] = {"attempt": attempt, **result.span_attributes}
            if wait_ms:
                attributes["wait_ms"] = round(wait_ms, 3)
            trace.spans.append(
                Span(
                    kind="generation",
                    name=f"{self.sut.name}.invoke",
                    started_at=span_start,
                    latency_ms=max(0.0, timer.latency_ms - wait_ms),
                    attributes=attributes,
                    error=result.error,
                )
            )
            trace.final_answer = result.answer
            trace.retrieved_doc_ids = list(result.retrieved_doc_ids)
            trace.tool_calls = list(result.tool_calls)
            # A failure reported by the system under test (e.g. a provider
            # error) becomes the trace error, so the taxonomy classifies it as
            # invocation_error rather than as an empty answer.
            trace.error = result.error
            trace.usage = self._usage(result)
            break
        return trace

    def _usage(self, result: SUTResult) -> Usage:
        """Token usage for one completed attempt.

        Counts reported by the SUT are kept as given. A failed call reports
        nothing and gets token_source "none" with zero tokens; it is never
        filled with an estimate. Otherwise tokens are estimated from the
        prompt and answer text and labelled "heuristic".
        """
        if result.usage:
            return Usage(
                prompt_tokens=int(result.usage.get("prompt_tokens") or 0),
                completion_tokens=int(result.usage.get("completion_tokens") or 0),
                token_source=str(result.usage.get("token_source", "api")),
            )
        if result.error or not self.config.heuristic_tokens:
            return Usage(token_source="none")
        return Usage(
            prompt_tokens=estimate_tokens(result.prompt_text),
            completion_tokens=estimate_tokens(result.answer or ""),
            token_source="heuristic",
        )
