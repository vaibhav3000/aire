"""Turn repo-engineer agent logs into aire traces.

repo-engineer writes one record per task: either one JSON object per file
(results/benchmark_traces/<task>_trace.json) or one object per line (JSONL).
Each record pairs `actions` (tool, args, step) with `observations` (ok,
rejected, error, exit_code) and ends in a `state` and a `verified` flag.

Mapping:
- task_id becomes the case id; every action becomes a tool call
  {name, args, step, ok, valid, error}, where valid is False when the
  runtime rejected the call (schema or policy);
- a run that did not end COMPLETE and verified becomes an invocation error
  carrying the state and the agent's own error messages;
- llm_usage (calls, prompt_tokens, completion_tokens) becomes API token
  usage; a scripted run has none and gets token_source "none";
- the logs carry no timings, so no spans are invented and the engine reports
  no latency for these traces.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ..schema import Trace, Usage, validate_trace_dict


def _records(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                yield f"{path.name}:{line_no}", json.loads(line)
        return
    data = json.loads(text)
    items = data if isinstance(data, list) else [data]
    for i, item in enumerate(items):
        yield f"{path.name}[{i}]" if isinstance(data, list) else path.name, item


def _tool_calls(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = record.get("actions") or []
    observations = record.get("observations") or []
    calls = []
    for i, action in enumerate(actions):
        observation = observations[i] if i < len(observations) else {}
        calls.append(
            {
                "name": str(action.get("tool")),
                "args": dict(action.get("args") or {}),
                "step": action.get("step"),
                "ok": observation.get("ok"),
                "valid": not observation.get("rejected", False),
                "error": observation.get("error"),
            }
        )
    return calls


def record_to_trace(record: Mapping[str, Any], *, run_id: str, where: str = "") -> Trace:
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"{where}: repo-engineer record has no task_id")
    state = str(record.get("state", "UNKNOWN"))
    verified = bool(record.get("verified"))
    files = list(record.get("files_changed") or [])
    error = None
    if state != "COMPLETE" or not verified:
        messages = "; ".join(str(e)[:200] for e in record.get("errors") or [])
        error = f"agent ended in state {state} (verified={verified})" + (
            f": {messages}" if messages else ""
        )
    usage_data = record.get("llm_usage")
    if isinstance(usage_data, Mapping):
        usage = Usage(
            prompt_tokens=int(usage_data.get("prompt_tokens") or 0),
            completion_tokens=int(usage_data.get("completion_tokens") or 0),
            token_source="api",
        )
    else:
        usage = Usage(token_source="none")
    trace = Trace(
        run_id=run_id,
        case_id=task_id,
        system_config={
            "source": "repo-engineer",
            "model": usage_data.get("model") if isinstance(usage_data, Mapping) else "scripted",
        },
        started_at=0.0,
        final_answer=f"{state}: changed {', '.join(files) if files else 'nothing'}",
        tool_calls=_tool_calls(record),
        usage=usage,
        error=error,
        trace_id=f"tr_{task_id}",
    )
    problems = validate_trace_dict(trace.to_dict())
    if problems:
        raise ValueError(f"{where}: invalid trace: {problems}")
    return trace


def load_repo_engineer_traces(path: str | Path, *, run_id: str = "repo_engineer") -> list[Trace]:
    """Traces from one .json or .jsonl file, or every such file in a directory (by name)."""
    root = Path(path)
    files = (
        sorted(p for p in root.iterdir() if p.suffix in (".json", ".jsonl"))
        if root.is_dir()
        else [root]
    )
    traces = []
    for file in files:
        for where, record in _records(file):
            traces.append(record_to_trace(record, run_id=run_id, where=where))
    return traces
