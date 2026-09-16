"""TraceRecorder: convenience wrapper that builds a Trace while a system runs.

The runner builds traces from SUTResult objects; systems that want finer-grained
tracing (multiple spans: retrieval, rerank, generation, tool calls) use this
recorder directly. It enforces schema validation on finish, so invalid traces
fail loudly at the boundary instead of corrupting stored runs.
"""

from __future__ import annotations

import time

from ..schema import Span, Timer, Trace, validate_trace_dict


class TraceRecorder:
    """Collects spans for one invocation and emits a validated Trace."""

    def __init__(self, run_id: str, case_id: str, system_config: dict) -> None:
        self._trace = Trace(
            run_id=run_id,
            case_id=case_id,
            system_config=system_config,
            started_at=time.time(),
        )

    def span(self, kind: str, name: str) -> Timer:
        """Context manager: with rec.span("retrieval", "qdrant.search") as t: ..."""
        return SpanTimer(self, kind, name)

    def add_span(self, span: Span) -> None:
        self._trace.spans.append(span)

    def finish(
        self,
        final_answer: str | None,
        retrieved_doc_ids: list[str] | None = None,
        tool_calls: list[list | dict] | None = None,
        error: str | None = None,
    ) -> Trace:
        self._trace.final_answer = final_answer
        self._trace.retrieved_doc_ids = retrieved_doc_ids or []
        self._trace.tool_calls = tool_calls or []
        self._trace.error = error
        problems = validate_trace_dict(self._trace.to_dict())
        if problems:
            raise ValueError(f"invalid trace: {problems}")
        return self._trace

    @property
    def trace(self) -> Trace:
        return self._trace


class SpanTimer:
    """Binds a Timer to a span on the parent recorder."""

    def __init__(self, recorder: TraceRecorder, kind: str, name: str) -> None:
        self._recorder = recorder
        self._span = Span(
            kind=kind, name=name, started_at=time.time(), latency_ms=0.0
        )

    def __enter__(self) -> "SpanTimer":
        self._timer = Timer()
        self._timer.__enter__()
        return self

    @property
    def attributes(self) -> dict:
        """Attribute dict of the underlying span, so callers can annotate it."""
        return self._span.attributes

    def __exit__(self, exc_type, exc, tb) -> None:
        self._timer.__exit__(exc_type, exc, tb)
        self._span.latency_ms = self._timer.latency_ms
        if exc_type is not None:
            self._span.error = f"{exc_type.__name__}: {exc}"
        self._recorder.add_span(self._span)
        return False
