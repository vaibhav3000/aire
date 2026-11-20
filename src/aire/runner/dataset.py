"""Evaluation cases and datasets.

An EvalCase is the unit of evaluation: input to the system plus the
expectations the evaluator checks against the trace. Expectations are
declarative so the same dataset works for any system version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

_LIST_FIELDS = ("expected_keywords", "supporting_doc_ids", "tags", "expected_tool_sequence")
_BOOL_FIELDS = ("expect_abstention", "must_cite")
_OPTIONAL_STR_FIELDS = ("expected_answer", "expected_tool")


@dataclass
class EvalCase:
    case_id: str
    input: str  # user query / task input
    # --- expectations (all optional; the engine only checks what is present) ---
    expected_keywords: list[str] = field(default_factory=list)
    expected_answer: str | None = None  # for exact-match style checks
    supporting_doc_ids: list[str] = field(default_factory=list)  # retrieval ground truth
    expected_tool: str | None = None
    expected_tool_args: dict[str, Any] = field(default_factory=dict)
    # Reference trajectory for agent cases: the tool names an expert would
    # call, in order (evaluator.trajectory).
    expected_tool_sequence: list[str] = field(default_factory=list)
    expect_abstention: bool = False  # true for unanswerable questions
    must_cite: bool = True  # answer must reference supporting docs
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalCase:
        """Build a case from one JSON object; raise ValueError on any schema problem.

        Unknown fields are rejected rather than dropped, so a typo such as
        'expect_abstension' cannot silently turn an unanswerable case into an
        answerable one, and flags must be real booleans ('false' is truthy).
        """
        if not isinstance(d, dict):
            raise ValueError(f"expected a JSON object, got {type(d).__name__}")
        unknown = sorted(set(d) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown field(s): {', '.join(unknown)}")
        for name in ("case_id", "input"):
            value = d.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"field '{name}' must be a non-empty string")
        for name in _LIST_FIELDS:
            value = d.get(name, [])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"field '{name}' must be a list of strings")
        for name in _BOOL_FIELDS:
            if name in d and not isinstance(d[name], bool):
                raise ValueError(f"field '{name}' must be a boolean, got {d[name]!r}")
        for name in _OPTIONAL_STR_FIELDS:
            if d.get(name) is not None and not isinstance(d[name], str):
                raise ValueError(f"field '{name}' must be a string or null")
        if not isinstance(d.get("expected_tool_args", {}), dict):
            raise ValueError("field 'expected_tool_args' must be an object")
        return cls(**d)


def load_eval_dataset(path: str | Path) -> list[EvalCase]:
    """Load a JSONL evaluation set. One case per line; case ids must be unique."""
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = EvalCase.from_dict(json.loads(line))
            except ValueError as exc:  # includes json.JSONDecodeError
                raise ValueError(f"eval dataset line {line_no}: {exc}") from exc
            if case.case_id in seen:
                raise ValueError(f"eval dataset line {line_no}: duplicate case_id {case.case_id!r}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError(f"no evaluation cases found in {path}")
    return cases
