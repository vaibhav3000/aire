"""Ingesting repo-engineer agent logs as aire traces (JSON per file or JSON lines)."""

import json
from pathlib import Path

import pytest

from aire.agents.ingest import load_repo_engineer_traces
from aire.schema import validate_trace_dict

ROOT = Path(__file__).resolve().parents[1]


def _record(task_id: str, state: str = "COMPLETE", verified: bool = True) -> dict:
    return {
        "task_id": task_id,
        "task_description": "Fix the loop.",
        "state": state,
        "actions": [
            {"step": "analyze", "tool": "list_tree"},
            {"step": 1, "tool": "search", "args": {"pattern": "range"}},
            {
                "step": 2,
                "tool": "apply_edit",
                "args": {"path": "a.py", "old_text": "x", "new_text": "y"},
            },
        ],
        "observations": [
            {"tool": "list_tree", "ok": True},
            {"tool": "search", "ok": True, "rejected": False, "error": None, "exit_code": None},
            {
                "tool": "apply_edit",
                "ok": False,
                "rejected": True,
                "error": "protected path",
                "exit_code": None,
            },
        ],
        "errors": [] if verified else ["tests still fail"],
        "files_changed": ["a.py"],
        "steps_used": 2,
        "replans": 0,
        "verified": verified,
        "llm_usage": {"calls": 3, "prompt_tokens": 1200, "completion_tokens": 80},
    }


def test_a_json_file_becomes_one_valid_trace(tmp_path: Path) -> None:
    path = tmp_path / "t1_trace.json"
    path.write_text(json.dumps(_record("t1")), encoding="utf-8")
    (trace,) = load_repo_engineer_traces(path, run_id="run_x")
    assert validate_trace_dict(trace.to_dict()) == []
    assert trace.case_id == "t1"
    assert trace.run_id == "run_x"
    assert [c["name"] for c in trace.tool_calls] == ["list_tree", "search", "apply_edit"]
    assert trace.tool_calls[0]["args"] == {}
    assert trace.tool_calls[1]["valid"] is True
    assert trace.tool_calls[2]["valid"] is False  # the runtime rejected it
    assert trace.tool_calls[2]["error"] == "protected path"
    assert trace.error is None
    assert trace.final_answer and "COMPLETE" in trace.final_answer
    assert (trace.usage.prompt_tokens, trace.usage.completion_tokens) == (1200, 80)
    assert trace.usage.token_source == "api"
    assert trace.spans == []  # the log has no timings, so none are invented


def test_jsonl_and_directories_are_read_in_order(tmp_path: Path) -> None:
    (tmp_path / "b.jsonl").write_text(
        json.dumps(_record("t2")) + "\n" + json.dumps(_record("t3")) + "\n", encoding="utf-8"
    )
    (tmp_path / "a_trace.json").write_text(json.dumps(_record("t1")), encoding="utf-8")
    traces = load_repo_engineer_traces(tmp_path)
    assert [t.case_id for t in traces] == ["t1", "t2", "t3"]


def test_an_unverified_or_failed_run_is_an_invocation_error(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_record("t1", state="FAILED", verified=False)), encoding="utf-8")
    (trace,) = load_repo_engineer_traces(path)
    assert trace.error is not None and "FAILED" in trace.error and "tests still fail" in trace.error


def test_a_scripted_run_has_no_token_counts(tmp_path: Path) -> None:
    record = _record("t1")
    record["llm_usage"] = None
    path = tmp_path / "t.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    (trace,) = load_repo_engineer_traces(path)
    assert trace.usage.token_source == "none"


def test_a_record_without_a_task_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"actions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="task_id"):
        load_repo_engineer_traces(path)


def test_the_committed_real_trajectories_load() -> None:
    traces = load_repo_engineer_traces(ROOT / "demo" / "agent" / "traces")
    assert len(traces) == 4
    assert all(t.error is None for t in traces)
    assert all(t.tool_calls and t.tool_calls[-1]["name"] == "run_tests" for t in traces)
