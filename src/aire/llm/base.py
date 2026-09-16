"""System-under-test and judge interfaces.

AIRE evaluates *any* AI system that implements the small `SystemUnderTest`
protocol. Committed demo results use the deterministic responder so that every
number in this repository is reproducible without network access or API keys;
the OpenAI-compatible adapter exists for real LLMs and is opt-in.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
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
    error: str | None = None


class SystemUnderTest(abc.ABC):
    """A system under test. Implementations must be deterministic OR clearly
    declare nondeterminism via `deterministic = False`."""

    name: str = "sut"
    config: dict[str, Any] = {}
    deterministic: bool = True

    @abc.abstractmethod
    def invoke(self, case_input: str, case_context: dict[str, Any] | None = None) -> SUTResult:
        ...


class Judge(abc.ABC):
    """Scores one aspect of a trace. Deterministic judges compute from the
    trace alone; LLM judges may call a model but must be injected explicitly."""

    name: str = "judge"

    @abc.abstractmethod
    def score(self, trace: Trace, context: dict[str, Any]) -> dict[str, Any]:
        """Return {"score": float|bool|None, "detail": str} — score in [0,1] or bool."""
        ...
