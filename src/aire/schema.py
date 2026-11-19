"""Trace and evaluation schemas.

A *trace* is the complete, serialisable record of one system invocation for
one evaluation case. Everything downstream (metrics, judges, regression,
reports) reads traces; nothing re-runs the system.

Schema rules:
- Every field is JSON-serialisable.
- Spans are an ordered list; each span has a kind, timing and payload.
- Validation happens on write (TraceRecorder) so stored files are trusted.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any

# Written into every serialised trace. Readers ignore fields they do not
# know, so additive changes stay readable by older code.
TRACE_SCHEMA_VERSION = "1.2"


def new_id(prefix: str) -> str:
    """Short, sortable-enough unique id: <prefix>_<8 hex>."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Span:
    """One timed step inside a trace (retrieval, generation, tool call, ...)."""

    kind: str  # "retrieval" | "generation" | "tool_call" | "custom"
    name: str  # human-readable step name
    started_at: float  # unix seconds
    latency_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None  # non-null means the span failed
    # A root span (parent_id None) covers one attempt at the whole case; child
    # spans (model attempts, retrieval, tool calls) point at it. Ids are only
    # written when set, so flat traces keep their Step 1 shape.
    span_id: str | None = None
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("span_id", "parent_id"):
            if data[key] is None:
                del data[key]
        return data


@dataclass
class Usage:
    """Token accounting. Heuristic counters are allowed but must be labelled."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_source: str = "unknown"  # "api" | "heuristic" | "none"
    # Breakdowns as reported by the provider (None when not reported).
    # cache_read_tokens is part of prompt_tokens and reasoning_tokens part of
    # completion_tokens; neither is added again to the total.
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Trace:
    """Full record of one system invocation for one evaluation case."""

    run_id: str
    case_id: str
    system_config: dict[str, Any]  # SUT version/config fingerprint
    started_at: float = field(default_factory=time.time)
    final_answer: str | None = None
    retrieved_doc_ids: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    error: str | None = None  # non-null means the whole invocation failed
    trace_id: str = field(default_factory=lambda: new_id("tr"))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = asdict(self.usage)
        d["spans"] = [s.to_dict() if isinstance(s, Span) else s for s in self.spans]
        d["schema_version"] = TRACE_SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        span_fields = {f.name for f in fields(Span)}
        usage_fields = {f.name for f in fields(Usage)}
        spans = [
            Span(**{k: v for k, v in s.items() if k in span_fields}) for s in d.get("spans", [])
        ]
        usage = Usage(**{k: v for k, v in d.get("usage", {}).items() if k in usage_fields})
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

    def __enter__(self) -> Timer:
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
    tool_calls = d.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        problems.append("tool_calls must be a list")
    else:
        for i, call in enumerate(tool_calls):
            if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                problems.append(f"tool_calls[{i}]: must be an object with a string 'name'")
    return problems
