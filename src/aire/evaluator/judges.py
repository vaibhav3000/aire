"""Judges: scoring functions that go beyond deterministic string checks.

Design position (deliberate, defensible): AIRE ships ONE built-in deterministic
judge (citation groundedness) and a clean interface for model-based judges.
We do NOT oversell LLM-as-judge — the interface records the judge's own
identity and configuration in every result, and the interview docs discuss
position bias, verbosity bias, self-preference and cost. An LLM judge is a
measurement instrument that itself needs evaluation.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..llm.base import Judge
from ..schema import Trace
from .metrics import groundedness as lexical_groundedness


class GroundednessJudge(Judge):
    """Deterministic citation-groundedness judge (lexical overlap proxy).

    See evaluator.metrics.groundedness for the exact definition and its limits.
    """

    name = "groundedness_lexical"

    def __init__(self, corpus: dict[str, str]) -> None:
        self.corpus = corpus

    def score(self, trace: Trace, context: dict[str, Any]) -> dict[str, Any]:
        value = lexical_groundedness(trace.final_answer, self.corpus)
        detail = {
            None: "not measurable (empty or abstained answer)",
        }.get(value)
        if detail is None:
            detail = "lexical citation-overlap proxy; deterministic"
        return {"score": value, "detail": detail}


class LLMJudge(Judge):
    """Pluggable model-based judge. NOT used by committed demo results.

    The scorer receives the full trace dict and returns a score in [0,1].
    It is the integrator's responsibility to evaluate the judge itself
    (agreement with human labels) before trusting it; AIRE records judge
    metadata to make that audit possible.
    """

    name = "llm_judge"

    def __init__(self, scorer: Any, description: str, version: str = "unversioned") -> None:
        self.scorer = scorer  # callable(trace_dict, case_dict) -> float in [0,1]
        self.description = description
        self.version = version

    def score(self, trace: Trace, context: dict[str, Any]) -> dict[str, Any]:
        trace_dict = trace.to_dict()
        value = float(self.scorer(trace_dict, context))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"LLMJudge returned out-of-range score {value}")
        return {
            "score": value,
            "detail": {
                "judge": self.description,
                "version": self.version,
                "warning": "model-based judge; validate against human labels before trusting",
            },
        }


class RubricJudge(Judge):
    """Abstract base for rubric-based judges with an explicit rubric."""

    name = "rubric_judge"

    @abstractmethod
    def rubric(self) -> dict[str, Any]:
        ...

    def score(self, trace: Trace, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
