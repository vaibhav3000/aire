"""A PR that changes the evaluation set is reported, not failed, and still gated.

The base gate refuses to pair runs over different evaluation sets (exit 2).
With --report it then writes a "not_comparable" report, and `aire pr-check`
turns that one case into a warning while every other exit 2 and every
regression, against the base branch or against the PR's own committed
results, still fails the check.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from aire.ci.pr_check import decide
from aire.cli import main
from aire.regression.gate import EXIT_USAGE, run_gate
from aire.report.markdown import MARKER, render_gate_markdown

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "v2_improved"
OLD_SHA = "3aeffeec" + "0" * 56


def _base_on_an_older_eval_set(tmp_path: Path) -> Path:
    """The committed v2 run, relabelled as if the base branch had another eval set."""
    base = tmp_path / "base_v2"
    shutil.copytree(V2, base)
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    manifest["dataset_sha256"] = OLD_SHA
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return base


def _not_comparable(tmp_path: Path) -> tuple[int, Path]:
    report = tmp_path / "gate.json"
    code = run_gate([str(_base_on_an_older_eval_set(tmp_path)), str(V2), "--report", str(report)])
    return code, report


def test_a_changed_evaluation_set_exits_2_with_a_not_comparable_report(tmp_path: Path) -> None:
    code, report = _not_comparable(tmp_path)
    assert code == EXIT_USAGE  # the gate still refuses; it is not weakened
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "not_comparable"
    assert data["reason"] == "dataset_changed"
    current = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    assert data["dataset"] == {
        "baseline_path": current["dataset_path"],
        "baseline_sha256": OLD_SHA,
        "candidate_path": current["dataset_path"],
        "candidate_sha256": current["dataset_sha256"],
    }


def test_any_other_usage_error_leaves_no_report(tmp_path: Path) -> None:
    report = tmp_path / "gate.json"
    report.write_text('{"status": "not_comparable", "reason": "dataset_changed"}', "utf-8")
    code = run_gate([str(tmp_path / "missing"), str(V2), "--report", str(report)])
    assert code == EXIT_USAGE
    assert not report.exists()  # a stale report cannot pass for this run's


def _pr_check(tmp_path: Path, base_exit: str, self_exit: str) -> int:
    return main(["pr-check", str(tmp_path / "gate.json"), "--base-exit", base_exit,
                 "--self-exit", self_exit])  # fmt: skip


def test_a_changed_evaluation_set_passes_the_check_with_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, report = _not_comparable(tmp_path)
    assert _pr_check(tmp_path, str(code), "0") == 0
    out = capsys.readouterr().out
    assert "::warning title=Not comparable::this PR changes the evaluation set" in out
    assert OLD_SHA in out and "demo/eval_set.jsonl" in out
    assert json.loads(report.read_text(encoding="utf-8"))["self_gate"] == {"exit_code": 0}


def test_a_regression_against_the_prs_own_results_still_fails(tmp_path: Path) -> None:
    code, _ = _not_comparable(tmp_path)
    assert _pr_check(tmp_path, str(code), "1") == 1


@pytest.mark.parametrize(
    ("base_exit", "self_exit", "report", "expected"),
    [
        (0, 0, None, 0),
        (1, 0, None, 1),  # regressed against the base branch
        (0, 1, None, 1),  # regressed against the PR's own results
        (2, 0, None, 1),  # a crash: no report
        (2, 0, {"status": "not_comparable", "reason": "something_else", "dataset": {}}, 1),
        (2, 0, {"summary": {}, "metrics": []}, 1),  # an ordinary report is no excuse
        (None, 0, None, 1),  # the base gate never ran
        (0, None, None, 1),  # the self gate never ran
        (2, 2, {"status": "not_comparable", "reason": "dataset_changed", "dataset": {}}, 1),
    ],
)
def test_every_other_outcome_is_decided_as_before(
    base_exit: int | None, self_exit: int | None, report: Any, expected: int
) -> None:
    assert decide(base_exit, self_exit, report).exit_code == expected


def test_the_comment_names_the_changed_dataset_and_both_hashes(tmp_path: Path) -> None:
    _, report = _not_comparable(tmp_path)
    data = json.loads(report.read_text(encoding="utf-8"))
    data["self_gate"] = {"exit_code": 0}
    text = render_gate_markdown(data, title="v2_improved against the base branch")
    assert text.startswith(MARKER)
    assert "**Not comparable: this PR changes the evaluation set.**" in text
    assert "`demo/eval_set.jsonl`" in text
    assert f"`{OLD_SHA}`" in text and f"`{data['dataset']['candidate_sha256']}`" in text
    assert "gated against the results committed in the pull request instead: passed." in text


def test_the_comment_shows_a_failed_self_gate_and_escapes_untrusted_strings() -> None:
    report = {
        "status": "not_comparable",
        "reason": "dataset_changed",
        "dataset": {
            "candidate_path": "demo/x.jsonl <img src=x> @owner [a](b)",
            "baseline_sha256": "a" * 64,
            "candidate_sha256": "b" * 64,
        },
        "self_gate": {"exit_code": 1},
    }
    text = render_gate_markdown(report, title="t")
    assert "<img" not in text and "@owner" not in text and "](" not in text
    assert "**blocked**" in text


def test_a_not_comparable_report_without_its_dataset_is_refused() -> None:
    with pytest.raises(ValueError, match="dataset_changed"):
        render_gate_markdown({"status": "not_comparable", "reason": "x"}, title="t")
