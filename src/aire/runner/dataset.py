"""Evaluation cases and datasets.

An EvalCase is the unit of evaluation: input to the system plus the
expectations the evaluator checks against the trace. Expectations are
declarative so the same dataset works for any system version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    case_id: str
    input: str                                   # user query / task input
    # --- expectations (all optional; the engine only checks what is present) ---
    expected_keywords: list[str] = field(default_factory=list)
    expected_answer: str | None = None           # for exact-match style checks
    supporting_doc_ids: list[str] = field(default_factory=list)  # retrieval ground truth
    expected_tool: str | None = None
    expected_tool_args: dict[str, Any] = field(default_factory=dict)
    expect_abstention: bool = False              # true for unanswerable questions
    must_cite: bool = True                       # answer must reference supporting docs
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalCase":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def load_eval_dataset(path: str | Path) -> list[EvalCase]:
    """Load a JSONL evaluation set. One case per line."""
    cases: list[EvalCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(EvalCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"eval dataset line {line_no}: {exc}") from exc
    if not cases:
        raise ValueError(f"no evaluation cases found in {path}")
    return cases
