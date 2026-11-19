"""AsyncExperimentRunner: the runner for systems that call rate-limited models.

Same contract and run layout as ExperimentRunner (manifest.json plus
traces.jsonl, one validated trace per case), with three differences:

- Cases run concurrently on a fixed pool of worker tasks (`concurrency`).
  Pacing, retries and caching of model calls belong to the system's shared
  ModelClient, not to the runner.
- A timeout cancels an async system for real (asyncio.timeout), which closes
  its connections. Sync systems run in a worker thread through
  asyncio.to_thread and keep the old limitation that a hung call cannot be
  killed.
- Traces are appended and flushed as cases finish, so a crashed or
  interrupted process keeps every finished case, and the file is rewritten
  (and fsynced) in dataset order at the end, so its content does not depend
  on completion order. Fsyncing every line as well would block the event
  loop on each disk sync and serialise the workers; it survives power loss
  rather than process loss, and is available as fsync_each_trace.

Each attempt gets a root span (span_id set) whose latency excludes the waits
the system reports; the system's own spans (retrieval, model attempts) are
attached under it.
"""

from __future__ import annotations

import abc
import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..llm.base import SUTResult, SystemUnderTest
from ..schema import Span, Trace, Usage, new_id
from ..tracing.storage import RunStore
from .dataset import EvalCase, load_eval_dataset
from .runner import usage_from_result


class AsyncSystemUnderTest(abc.ABC):
    """A system under test whose invocation is a coroutine."""

    name: str = "sut"
    config: Mapping[str, Any] = MappingProxyType({})
    deterministic: bool = True

    @abc.abstractmethod
    async def ainvoke(
        self, case_input: str, case_context: dict[str, Any] | None = None
    ) -> SUTResult: ...

    async def aclose(self) -> None:
        """Release clients and connections; the runner calls it once at the end."""
        return None


@dataclass(frozen=True)
class AsyncRunnerConfig:
    concurrency: int = 8
    timeout_s: float = 120.0
    max_retries: int = 0  # whole-invocation retries; model-call retries live in ModelClient
    retry_backoff_s: float = 0.2
    heuristic_tokens: bool = True
    fsync_each_trace: bool = False  # see the module docstring


@dataclass(frozen=True)
class RunOutcome:
    run_dir: Path
    n_cases: int
    n_failed: int  # cases whose invocation failed or returned no answer
    wall_time_s: float
    cases_per_s: float


def _failed(trace: Trace) -> bool:
    return bool(trace.error) or not (trace.final_answer or "").strip()


class AsyncExperimentRunner:
    def __init__(
        self,
        sut: SystemUnderTest | AsyncSystemUnderTest,
        config: AsyncRunnerConfig | None = None,
    ) -> None:
        self.sut = sut
        self.config = config or AsyncRunnerConfig()
        if self.config.concurrency < 1:
            raise ValueError("concurrency must be at least 1")

    def run(
        self,
        dataset_path: str | Path,
        runs_root: str | Path = "runs",
        run_name: str | None = None,
        *,
        corpus: Mapping[str, str] | None = None,
        path_root: str | Path | None = None,
    ) -> RunOutcome:
        """Blocking wrapper around arun() for scripts and tests."""
        return asyncio.run(
            self.arun(dataset_path, runs_root, run_name, corpus=corpus, path_root=path_root)
        )

    async def arun(
        self,
        dataset_path: str | Path,
        runs_root: str | Path = "runs",
        run_name: str | None = None,
        *,
        corpus: Mapping[str, str] | None = None,
        path_root: str | Path | None = None,
    ) -> RunOutcome:
        cases: list[EvalCase] = load_eval_dataset(dataset_path)
        run_id = new_id("run")
        store = RunStore(
            runs_root,
            run_id,
            run_name or f"{self.sut.name}_{run_id}",
            fsync=self.config.fsync_each_trace,
        )
        store.write_manifest(
            created_at=time.time(),
            system_config=dict(self.sut.config),
            dataset_path=dataset_path,
            n_cases=len(cases),
            deterministic=self.sut.deterministic,
            corpus=corpus,
            path_root=path_root,
        )
        queue: asyncio.Queue[EvalCase] = asyncio.Queue()
        for case in cases:
            queue.put_nowait(case)
        failed = 0

        async def worker() -> None:
            nonlocal failed
            while True:
                try:
                    case = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                trace = await self._run_case(case, run_id)
                store.append_trace(trace)
                failed += int(_failed(trace))

        start = time.perf_counter()
        try:
            n_workers = min(self.config.concurrency, len(cases))
            async with asyncio.TaskGroup() as group:
                for _ in range(n_workers):
                    group.create_task(worker())
        finally:
            if isinstance(self.sut, AsyncSystemUnderTest):
                await self.sut.aclose()
        wall = time.perf_counter() - start
        store.rewrite_in_order([c.case_id for c in cases])
        return RunOutcome(
            run_dir=store.run_dir,
            n_cases=len(cases),
            n_failed=failed,
            wall_time_s=wall,
            cases_per_s=len(cases) / wall if wall > 0 else float("inf"),
        )

    async def _invoke(self, case_input: str) -> SUTResult:
        if isinstance(self.sut, AsyncSystemUnderTest):
            return await self.sut.ainvoke(case_input)
        sync_sut = self.sut
        await asyncio.to_thread(sync_sut.before_invoke)
        return await asyncio.to_thread(sync_sut.invoke, case_input)

    async def _run_case(self, case: EvalCase, run_id: str) -> Trace:
        trace = Trace(
            run_id=run_id,
            case_id=case.case_id,
            system_config=dict(self.sut.config),
            started_at=time.time(),
            usage=Usage(token_source="none"),
        )
        attempts = self.config.max_retries + 1
        span_name = f"{self.sut.name}.invoke"
        for attempt in range(1, attempts + 1):
            root_id = new_id("sp")
            started_at = time.time()
            start = time.perf_counter()
            try:
                async with asyncio.timeout(self.config.timeout_s):
                    result = await self._invoke(case.input)
            except Exception as exc:  # timeout or system failure: recorded, retried, classified
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if isinstance(exc, TimeoutError):
                    error = (
                        f"InvocationTimeout: invocation exceeded {self.config.timeout_s:g}s timeout"
                    )
                else:
                    error = f"{type(exc).__name__}: {exc}"
                trace.spans.append(
                    Span(
                        kind="generation",
                        name=span_name,
                        started_at=started_at,
                        latency_ms=elapsed_ms,
                        attributes={"attempt": attempt},
                        error=error,
                        span_id=root_id,
                    )
                )
                trace.error = error
                if attempt < attempts:
                    await asyncio.sleep(self.config.retry_backoff_s * attempt)
                continue

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            wait_ms = max(0.0, float(result.wait_ms))
            attributes: dict[str, Any] = {"attempt": attempt, **result.span_attributes}
            if wait_ms:
                attributes["wait_ms"] = round(wait_ms, 3)
            trace.spans.append(
                Span(
                    kind="generation",
                    name=span_name,
                    started_at=started_at,
                    latency_ms=max(0.0, elapsed_ms - wait_ms),
                    attributes=attributes,
                    error=result.error,
                    span_id=root_id,
                )
            )
            for child in result.spans:
                if child.parent_id is None:
                    child.parent_id = root_id
                trace.spans.append(child)
            trace.final_answer = result.answer
            trace.retrieved_doc_ids = list(result.retrieved_doc_ids)
            trace.tool_calls = list(result.tool_calls)
            trace.error = result.error
            trace.usage = usage_from_result(result, heuristic_tokens=self.config.heuristic_tokens)
            break
        return trace
