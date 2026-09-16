"""Trace and evaluation schemas.

A *trace* is the complete, serialisable record of one system invocation for
one evaluation case. Everything downstream (metrics, judges, regression,
reports) reads traces — nothing re-runs the system.

Schema rules:
- Every field is JSON-serialisable.
- Spans are an ordered list; each span has a kind, timing and payload.
- Validation happens on write (TraceRecorder) so stored files are trusted.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def new_id(prefix: str) -> str:
    """Short, sortable-enough unique id: <prefix>_<8 hex>."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Span:
    """One timed step inside a trace (retrieval, generation, tool call, ...)."""

    kind: str                      # "retrieval" | "generation" | "tool_call" | "custom"
    name: str                      # human-readable step name
    started_at: float              # unix seconds
    latency_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None       # non-null means the span failed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Usage:
    """Token accounting. Heuristic counters are allowed but must be labelled."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_source: str = "unknown"  # "api" | "heuristic" | "none"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Trace:
    """Full record of one system invocation for one evaluation case."""

    run_id: str
    case_id: str
    system_config: dict[str, Any]      # SUT version/config fingerprint
    started_at: float = field(default_factory=time.time)
    final_answer: str | None = None
    retrieved_doc_ids: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    error: str | None = None           # non-null means the whole invocation failed
    trace_id: str = field(default_factory=lambda: new_id("tr"))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = asdict(self.usage)
        d["spans"] = [s.to_dict() if isinstance(s, Span) else s for s in self.spans]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trace":
        spans = [Span(**s) for s in d.get("spans", [])]
        usage = Usage(**d.get("usage", {}))
        return cls(
            run_id=d["run_id"],
            case_id=d["case_id"],
            system_config=d.get("system_config", {}),
            started_at=d.get("started_at", 0.0),
            final_answer=d.get("final_answer"),
            retrieved_doc_ids=d.get("retrieved_doc_ids", []),
            tool_calls=d.get("tool_calls", []),
            spans=spans,
            usage=usage,
            error=d.get("error"),
            trace_id=d.get("trace_id", new_id("tr")),
        )


def estimate_tokens(text: str) -> int:
    """Heuristic token estimate (~4 chars/token). Labelled 'heuristic' in Usage."""
    return max(1, len(text) // 4) if text else 0


class Timer:
    """Context manager measuring wall-clock latency in milliseconds."""

    def __init__(self) -> None:
        self.start: float = 0.0
        self.latency_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.latency_ms = (time.perf_counter() - self.start) * 1000


def validate_trace_dict(d: dict[str, Any]) -> list[str]:
    """Lightweight structural validation; returns a list of problems (empty = valid)."""
    problems: list[str] = []
    for key in ("run_id", "case_id", "trace_id"):
        if not d.get(key):
            problems.append(f"missing required field: {key}")
    for i, span in enumerate(d.get("spans", [])):
        if "kind" not in span:
            problems.append(f"span[{i}]: missing 'kind'")
        if "latency_ms" not in span:
            problems.append(f"span[{i}]: missing 'latency_ms'")
        if span.get("latency_ms") is not None and span["latency_ms"] < 0:
            problems.append(f"span[{i}]: negative latency_ms")
    u = d.get("usage", {})
    if u.get("prompt_tokens", 0) < 0 or u.get("completion_tokens", 0) < 0:
        problems.append("usage tokens must be non-negative")
    return problems
