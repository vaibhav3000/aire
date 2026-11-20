"""The outcome of the PR evaluation check, from the gates' exit codes.

eval-pr.yml runs two gates on a pull request:

- the base gate compares the PR's fresh run with the base branch's committed
  results;
- the self gate compares the PR's fresh run with the results committed in the
  PR itself, as the CI eval-gate job does.

A base gate that exits 2 because the PR changes the evaluation set is not a
failure: the runs cannot be paired, so the check reports "not comparable" and
relies on the self gate. Any other exit 2 (a crash, a missing or malformed
run) and any regression still fails the check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

EXIT_PASS, EXIT_REGRESSED, EXIT_USAGE = 0, 1, 2
_UNSAFE = re.compile(r"[^A-Za-z0-9 _.,:;=+\-/()#']")


def _safe(value: Any, limit: int = 120) -> str:
    return _UNSAFE.sub("_", str(value))[:limit]


def dataset_change(report: Any) -> dict[str, str] | None:
    """The dataset paths and hashes of a "not comparable" gate report, else None."""
    if not isinstance(report, dict):
        return None
    if report.get("status") != "not_comparable" or report.get("reason") != "dataset_changed":
        return None
    dataset = report.get("dataset")
    if not isinstance(dataset, dict):
        return None
    keys = ("baseline_path", "baseline_sha256", "candidate_path", "candidate_sha256")
    return {key: _safe(dataset.get(key)) for key in keys}


@dataclass
class CheckOutcome:
    """The check's exit code and the GitHub annotations that explain it."""

    exit_code: int
    comparable: bool
    annotations: list[str] = field(default_factory=list)


def decide(base_exit: int | None, self_exit: int | None, base_report: Any) -> CheckOutcome:
    """Decide the PR check. A missing exit code means the gate never ran."""
    annotations: list[str] = []
    failed = False
    change = dataset_change(base_report) if base_exit == EXIT_USAGE else None
    if change is not None:
        annotations.append(
            "::warning title=Not comparable::this PR changes the evaluation set "
            f"({change['candidate_path']}: sha256 {change['baseline_sha256']} on the base "
            f"branch, {change['candidate_sha256']} here), so it was not gated against the "
            "base branch; the fresh run was gated against the PR's own committed results"
        )
    elif base_exit != EXIT_PASS:
        failed = True
        what = "found a regression" if base_exit == EXIT_REGRESSED else f"exited {base_exit}"
        annotations.append(f"::error::the gate against the base branch {what}")
    if self_exit != EXIT_PASS:
        failed = True
        what = "found a regression" if self_exit == EXIT_REGRESSED else f"exited {self_exit}"
        annotations.append(f"::error::the gate against the PR's own committed results {what}")
    return CheckOutcome(int(failed), comparable=change is None, annotations=annotations)
