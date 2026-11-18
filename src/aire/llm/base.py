"""System-under-test and judge interfaces.

AIRE evaluates *any* AI system that implements the small `SystemUnderTest`
protocol. Committed demo results use the deterministic responder so that every
number in this repository is reproducible without network access or API keys;
the OpenAI-compatible adapter exists for real LLMs and is opt-in.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..schema import Trace


@dataclass
class SUTResult:
    """What a system returns for one input; the runner wraps it into a Trace."""

    answer: str | None = None
    retrieved_doc_ids: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Optional per-step detail consumed by the runner for spans/usage:
    prompt_text: str = ""
    usage: dict[str, Any] | None = None  # {"prompt_tokens","completion_tokens","token_source"}
    error: str | None = None
    # Time spent deliberately waiting inside invoke() (e.g. rate-limit backoff
    # between provider retries). The runner records it on the span and
    # subtracts it from the measured latency.
    wait_ms: float = 0.0
    # Extra span attributes, e.g. the number of provider attempts.
    span_attributes: dict[str, Any] = field(default_factory=dict)


class SystemUnderTest(abc.ABC):
    """A system under test. Implementations must be deterministic OR clearly
    declare nondeterminism via `deterministic = False`."""

    name: str = "sut"
    # Read-only shared default; implementations assign their own dict per instance.
    config: Mapping[str, Any] = MappingProxyType({})
    deterministic: bool = True

    def before_invoke(self) -> None:
        """Hook the runner calls before each timed attempt, outside the timer.

        Use it for client-side pacing (waiting for a rate-limit slot), so the
        recorded latency measures the system rather than the rate limiter.
        The default does nothing.
        """
        return None

    @abc.abstractmethod
    def invoke(self, case_input: str, case_context: dict[str, Any] | None = None) -> SUTResult: ...


class Judge(abc.ABC):
    """Scores one aspect of a trace. Deterministic judges compute from the
    trace alone; LLM judges may call a model but must be injected explicitly."""

    name: str = "judge"

    @abc.abstractmethod
    def score(self, trace: Trace, context: dict[str, Any]) -> dict[str, Any]:
        """Return {"score": float|bool|None, "detail": str | dict}."""
        ...
