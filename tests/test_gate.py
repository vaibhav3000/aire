"""Tests for the CI regression gate (python -m aire.regression.gate).

The fixtures are the committed, evaluated demo runs in results/: v2 improves
retrieval and holds everything else; v3 drops citations (the seeded regression).
"""

import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

from aire.regression.gate import EXIT_PASS, EXIT_REGRESSED, EXIT_USAGE, run_gate

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
    assert {"groundedness", "citation_coverage"} <= regressed


def test_ignored_metrics_never_fail_the_gate() -> None:
    args = [
        V1,
        V3,
        "--ignore",
        "groundedness",
        "--ignore",
        "citation_coverage",
        "--ignore",
        "latency_ms",
    ]
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
