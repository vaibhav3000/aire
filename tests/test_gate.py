"""Tests for the CI regression gate (python -m aire.regression.gate).

The fixtures are the committed, evaluated demo runs in results/: v2 retrieves
three documents instead of one and holds everything else; v3 drops citations
(the seeded regression).
"""

import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

from aire.evaluator.engine import EvaluationEngine
from aire.regression.gate import EXIT_PASS, EXIT_REGRESSED, EXIT_USAGE, run_gate
from aire.runner.dataset import EvalCase
from aire.schema import Trace

ROOT = Path(__file__).resolve().parents[1]
V1 = str(ROOT / "results" / "v1_baseline")
V2 = str(ROOT / "results" / "v2_improved")
V3 = str(ROOT / "results" / "v3_fluent_regression")


def test_gate_passes_when_nothing_regresses() -> None:
    assert run_gate([V1, V2, "--ignore", "latency_ms"]) == EXIT_PASS


def test_gate_blocks_the_seeded_grounding_regression(tmp_path: Path) -> None:
    report = tmp_path / "gate.json"
    code = run_gate([V1, V3, "--ignore", "latency_ms", "--report", str(report)])
    assert code == EXIT_REGRESSED
    data = json.loads(report.read_text(encoding="utf-8"))
    regressed = {row["metric"] for row in data["metrics"] if row["verdict"] == "regressed"}
    assert {"groundedness", "gold_citation_recall"} <= regressed


def test_ignored_metrics_never_fail_the_gate() -> None:
    # Every metric v3 regresses on, ignored: the gate must pass.
    ignored = [
        "groundedness",
        "gold_citation_recall",
        "citation_precision",
        "success_rate",
        "latency_ms",
    ]
    args = [V1, V3]
    for metric in ignored:
        args += ["--ignore", metric]
    assert run_gate(args) == EXIT_PASS


def test_missing_run_directory_is_a_usage_error(tmp_path: Path) -> None:
    assert run_gate([str(tmp_path / "missing_run"), V2]) == EXIT_USAGE


def test_malformed_override_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        run_gate([V1, V2, "--override", "latency_ms"])
    assert exc.value.code == EXIT_USAGE


def test_installed_console_script_gates_the_committed_runs() -> None:
    """The aire-gate entry point (not just run_gate) must be installed and work."""
    exe = shutil.which("aire-gate", path=sysconfig.get_path("scripts"))
    assert exe is not None, "aire-gate is not installed; run pip install -e ."
    passed = subprocess.run(
        [exe, V1, V2, "--ignore", "latency_ms"], capture_output=True, text=True, check=False
    )
    assert passed.returncode == EXIT_PASS, passed.stdout + passed.stderr
    blocked = subprocess.run(
        [exe, V1, V3, "--ignore", "latency_ms"], capture_output=True, text=True, check=False
    )
    assert blocked.returncode == EXIT_REGRESSED, blocked.stdout + blocked.stderr
    assert "GATE FAILED" in blocked.stdout


def _write_results(run_dir: Path, results: str, sha: str = "a" * 64) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_results.json").write_text(results, encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({"dataset_sha256": sha}), encoding="utf-8")
    return str(run_dir)


_GOOD = json.dumps({"per_case": [{"case_id": "c1"}], "aggregates": {"mean_keyword_recall": 1.0}})


def test_malformed_results_are_a_usage_error_not_a_regression(tmp_path: Path) -> None:
    base = _write_results(tmp_path / "base", _GOOD)
    cand = _write_results(tmp_path / "cand", '{"per_case": [')
    assert run_gate([base, cand]) == EXIT_USAGE


def test_results_without_case_ids_are_a_usage_error(tmp_path: Path) -> None:
    base = _write_results(tmp_path / "base", _GOOD)
    cand = _write_results(tmp_path / "cand", json.dumps({"per_case": [{}], "aggregates": {}}))
    assert run_gate([base, cand]) == EXIT_USAGE


def test_runs_on_different_datasets_are_a_usage_error(tmp_path: Path) -> None:
    base = _write_results(tmp_path / "base", _GOOD)
    cand = _write_results(tmp_path / "cand", _GOOD, sha="b" * 64)
    assert run_gate([base, cand]) == EXIT_USAGE


def test_candidate_whose_unanswerable_cases_crash_is_blocked(tmp_path: Path) -> None:
    corpus = {"doc_a": "The refund window is 30 days for annual plans."}
    cases = [
        EvalCase(
            case_id="refund",
            input="How long is the refund window?",
            expected_keywords=["30 days"],
            supporting_doc_ids=["doc_a"],
        ),
        EvalCase(case_id="ipv6", input="IPv6?", expect_abstention=True, must_cite=False),
    ]
    answer = "The refund window is 30 days for annual plans [doc:doc_a]."
    refund = Trace(
        run_id="r",
        case_id="refund",
        system_config={},
        final_answer=answer,
        retrieved_doc_ids=["doc_a"],
    )
    abstains = Trace(run_id="r", case_id="ipv6", system_config={}, final_answer="I don't know.")
    crashes = Trace(run_id="r", case_id="ipv6", system_config={}, error="ProviderError: HTTP 429")
    engine = EvaluationEngine(corpus=corpus)
    base = _write_results(
        tmp_path / "base", json.dumps(engine.evaluate_run([refund, abstains], cases))
    )
    cand = _write_results(
        tmp_path / "cand", json.dumps(engine.evaluate_run([refund, crashes], cases))
    )
    report = tmp_path / "gate.json"
    assert run_gate([base, cand, "--report", str(report)]) == EXIT_REGRESSED
    data = json.loads(report.read_text(encoding="utf-8"))
    regressed = {row["metric"] for row in data["metrics"] if row["verdict"] == "regressed"}
    assert {"abstention_correct", "errors", "success_rate"} <= regressed


def test_gate_blocks_the_seeded_agent_trajectory_defects(tmp_path: Path) -> None:
    report = tmp_path / "gate.json"
    base = str(ROOT / "results" / "agent_baseline")
    cand = str(ROOT / "results" / "agent_regressed")
    assert run_gate([base, cand, "--report", str(report)]) == EXIT_REGRESSED
    data = json.loads(report.read_text(encoding="utf-8"))
    regressed = {row["metric"] for row in data["metrics"] if row["verdict"] == "regressed"}
    assert {"tool_args_schema_valid", "tool_selection_accuracy"} <= regressed
    assert run_gate([base, base]) == EXIT_PASS
