"""Trajectory metrics: tool choice against a reference, efficiency, schema-valid arguments."""

import pytest

from aire.evaluator.engine import EvaluationEngine
from aire.evaluator.metrics import metric_info
from aire.evaluator.trajectory import (
    lcs_length,
    step_efficiency,
    tool_args_schema_valid,
    tool_call_precision,
    tool_selection_accuracy,
)
from aire.runner.dataset import EvalCase
from aire.schema import Trace

REFERENCE = ["list_tree", "search", "read_file", "apply_edit", "run_tests"]
SCHEMAS = {
    "search": {
        "type": "object",
        "properties": {"pattern": {"type": "string", "maxLength": 512}},
        "required": ["pattern"],
        "additionalProperties": False,
    },
    "run_tests": {
        "type": "object",
        "properties": {"cmd": {"type": "string"}},
        "required": ["cmd"],
        "additionalProperties": False,
    },
}


def test_lcs_counts_in_order_matches_only() -> None:
    assert lcs_length(["a", "b", "c"], ["a", "c"]) == 2
    assert lcs_length(["c", "a"], ["a", "c"]) == 1
    assert lcs_length([], ["a"]) == 0


def test_a_trajectory_that_follows_the_reference_scores_one() -> None:
    assert tool_selection_accuracy(REFERENCE, REFERENCE) == 1.0
    assert tool_call_precision(REFERENCE, REFERENCE) == 1.0
    assert step_efficiency(REFERENCE, REFERENCE) == 1.0


def test_detours_cost_precision_and_efficiency_but_not_selection() -> None:
    actual = ["list_tree", "search", "search", "read_file", "read_file", "apply_edit", "run_tests"]
    assert tool_selection_accuracy(actual, REFERENCE) == 1.0
    assert tool_call_precision(actual, REFERENCE) == pytest.approx(5 / 7)
    assert step_efficiency(actual, REFERENCE) == pytest.approx(5 / 7)


def test_a_missing_tool_lowers_selection_accuracy() -> None:
    actual = ["list_tree", "apply_edit", "run_tests"]  # never looked at the code
    assert tool_selection_accuracy(actual, REFERENCE) == pytest.approx(3 / 5)
    assert step_efficiency(actual, REFERENCE) == 1.0  # shorter is not penalised


def test_an_empty_trajectory_scores_zero() -> None:
    assert tool_selection_accuracy([], REFERENCE) == 0.0
    assert tool_call_precision([], REFERENCE) == 0.0
    assert step_efficiency([], REFERENCE) == 0.0


def test_arguments_are_checked_against_each_tools_schema() -> None:
    calls = [
        {"name": "search", "args": {"pattern": "TODO"}},
        {"name": "search", "args": {"pattern": 7}},  # wrong type
        {"name": "run_tests", "args": {"cmd": "pytest", "shell": True}},  # unknown argument
        {"name": "delete_repo", "args": {}},  # not a declared tool
    ]
    assert tool_args_schema_valid(calls, SCHEMAS) == pytest.approx(1 / 4)


def _agent_trace(calls: list[dict]) -> Trace:
    return Trace(
        run_id="r", case_id="t1", system_config={}, final_answer="COMPLETE", tool_calls=calls
    )


def test_the_engine_scores_trajectories_and_blames_schema_errors_on_tools() -> None:
    case = EvalCase(
        case_id="t1",
        input="fix the bug",
        expected_tool_sequence=["search", "run_tests"],
        must_cite=False,
    )
    engine = EvaluationEngine(tool_schemas=SCHEMAS)
    good = engine.evaluate_case(
        _agent_trace(
            [
                {"name": "search", "args": {"pattern": "x"}, "valid": True},
                {"name": "run_tests", "args": {"cmd": "pytest"}, "valid": True},
            ]
        ),
        case,
    )
    assert good["tool_selection_accuracy"] == 1.0
    assert good["tool_call_precision"] == 1.0
    assert good["step_efficiency"] == 1.0
    assert good["tool_args_schema_valid"] == 1.0
    assert good["n_steps"] == 2
    assert good["failure"] is None

    bad = engine.evaluate_case(
        _agent_trace([{"name": "search", "args": {"pattern": 3}, "valid": True}]), case
    )
    assert bad["tool_args_schema_valid"] == 0.0
    assert bad["failure"] == "tool_error"


def test_the_new_metrics_are_registered() -> None:
    for name in (
        "tool_selection_accuracy",
        "tool_call_precision",
        "step_efficiency",
        "tool_args_schema_valid",
    ):
        info = metric_info(name)
        assert info is not None and info["direction"] == "higher" and info["bounded"]
    steps = metric_info("n_steps")
    assert steps is not None and steps["direction"] == "lower" and not steps["bounded"]


def test_a_trace_without_spans_has_no_latency() -> None:
    case = EvalCase(case_id="t1", input="q", must_cite=False)
    row = EvaluationEngine().evaluate_case(_agent_trace([]), case)
    assert row["latency_ms"] is None
