"""Tool-call quality metrics.

Tool-using systems are evaluated on: (1) did it call the expected tool,
(2) were the arguments schema-valid, (3) did the arguments carry the expected
values. Failures include malformed calls recorded by the runtime — a tool call
that never executed because it violated the schema is still a tool failure of
the system under test.
"""

from __future__ import annotations

from typing import Any


def tool_selection_correct(tool_calls: list[dict[str, Any]], expected_tool: str | None) -> float | None:
    """1.0 if the expected tool was called; None when the case has no tool expectation.

    Extra calls are not penalized here (arg correctness and traces cover that);
    this metric answers only "did the system use the right tool".
    """
    if expected_tool is None:
        return None
    return float(any(c.get("name") == expected_tool for c in tool_calls))


def tool_args_valid(tool_calls: list[dict[str, Any]]) -> float:
    """Fraction of executed-or-rejected tool calls whose args passed schema validation.

    Each recorded call is expected to carry {"name", "args", "valid": bool} —
    the runner records schema validation results, so a call the runtime
    rejected still counts against the system.
    """
    if not tool_calls:
        return 1.0
    valid = sum(1 for c in tool_calls if c.get("valid", False))
    return valid / len(tool_calls)


def tool_args_match(tool_calls: list[dict[str, Any]], expected_tool: str | None, expected_args: dict[str, Any]) -> float | None:
    """Fraction of expected arg key/value pairs present in the first call of the expected tool."""
    if expected_tool is None or not expected_args:
        return None
    call = next((c for c in tool_calls if c.get("name") == expected_tool), None)
    if call is None:
        return 0.0
    args = call.get("args", {})
    hits = sum(1 for k, v in expected_args.items() if args.get(k) == v)
    return hits / len(expected_args)
