"""Derive a seeded-defect copy of the real repo-engineer trajectories.

Each task's log gets exactly one documented mutation, so the trajectory
metrics and the gate can be shown catching agent regressions on known
defects. The output is synthetic and says so in every record
("synthetic_mutation"). Deterministic: the same input gives the same output.

    python demo/agent/make_variants.py
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "traces"
TARGET = HERE / "variants"

Record = dict[str, Any]


def _index_of(record: Record, tool: str, nth: int = 0) -> int:
    hits = [i for i, a in enumerate(record["actions"]) if a.get("tool") == tool]
    return hits[nth]


def string_line_number(record: Record) -> str:
    i = _index_of(record, "read_file")
    record["actions"][i]["args"]["start_line"] = "1"  # a string where the schema wants an integer
    return "read_file called with start_line as a string"


def repeated_search(record: Record) -> str:
    i = _index_of(record, "search")
    record["actions"].insert(i + 1, copy.deepcopy(record["actions"][i]))
    record["observations"].insert(i + 1, copy.deepcopy(record["observations"][i]))
    return "the same search issued twice in a row"


def skipped_test_run(record: Record) -> str:
    i = _index_of(record, "run_tests")  # the agent's own run, before the runtime's verify
    del record["actions"][i]
    del record["observations"][i]
    return "the agent's own test run before verification removed"


def unknown_argument(record: Record) -> str:
    i = _index_of(record, "apply_edit")
    record["actions"][i]["args"]["dry_run"] = True  # not a declared argument
    return "apply_edit called with an undeclared dry_run argument"


MUTATIONS: dict[str, Callable[[Record], str]] = {
    "task_01_fix_off_by_one": string_line_number,
    "task_02_failing_test": repeated_search,
    "task_03_add_missing_test": skipped_test_run,
    "task_04_constrained_refactor": unknown_argument,
}


def main() -> None:
    TARGET.mkdir(exist_ok=True)
    for path in sorted(SOURCE.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        mutate = MUTATIONS[record["task_id"]]
        record["synthetic_mutation"] = mutate(record)
        out = TARGET / path.name
        out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"{record['task_id']}: {record['synthetic_mutation']}")


if __name__ == "__main__":
    main()
