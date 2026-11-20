"""Trajectory metrics for tool-using agents.

A trajectory is the ordered list of tool names an agent called. A reference
trajectory is the sequence an expert would use, written from the task. The
comparison uses the longest common subsequence (LCS), so the agent gets
credit for the reference's tool choices made in the reference's order, while
detours and repeated calls cost precision and efficiency:

- tool_selection_accuracy = LCS / len(reference)
- tool_call_precision     = LCS / len(actual)
- step_efficiency         = min(1, len(reference) / len(actual))
- tool_args_schema_valid  = share of calls whose arguments validate against
                            the tool's JSON Schema (unknown tools are invalid)

An empty trajectory scores 0 on all three sequence metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..jsonschema import validate


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence of two name sequences."""
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def tool_selection_accuracy(actual: Sequence[str], reference: Sequence[str]) -> float:
    if not reference:
        return 1.0
    return lcs_length(actual, reference) / len(reference)


def tool_call_precision(actual: Sequence[str], reference: Sequence[str]) -> float:
    if not actual:
        return 0.0
    return lcs_length(actual, reference) / len(actual)


def step_efficiency(actual: Sequence[str], reference: Sequence[str]) -> float:
    if not actual:
        return 0.0
    return min(1.0, len(reference) / len(actual))


def tool_args_schema_valid(
    tool_calls: Sequence[Mapping[str, Any]], schemas: Mapping[str, Mapping[str, Any]]
) -> float:
    """Share of calls whose arguments satisfy their tool's schema (1.0 for no calls)."""
    if not tool_calls:
        return 1.0
    valid = 0
    for call in tool_calls:
        schema = schemas.get(str(call.get("name")))
        if schema is not None and not validate(call.get("args", {}), schema):
            valid += 1
    return valid / len(tool_calls)
