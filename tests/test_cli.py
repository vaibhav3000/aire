"""The aire command: gate, report, markdown, pr-comment, otel-export, otel-ingest."""

import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pytest

from aire.cli import main
from aire.tracing.storage import load_run_traces

ROOT = Path(__file__).resolve().parents[1]
V1 = str(ROOT / "results" / "v1_baseline")
V2 = str(ROOT / "results" / "v2_improved")
V3 = str(ROOT / "results" / "v3_fluent_regression")


def test_gate_passes_through_to_aire_gate() -> None:
    assert main(["gate", V1, V2, "--ignore", "latency_ms"]) == 0
    assert main(["gate", V1, V3, "--ignore", "latency_ms"]) == 1


def test_report_and_markdown_from_two_runs(tmp_path: Path) -> None:
    gate_json = tmp_path / "gate.json"
    assert main(["gate", V1, V3, "--ignore", "latency_ms", "--report", str(gate_json)]) == 1
    html = tmp_path / "diff.html"
    assert (
        main(["report", V1, V3, "--ignore", "latency_ms", "--out", str(html), "--title", "v3"]) == 0
    )
    assert "Blocked" in html.read_text(encoding="utf-8")
    md = tmp_path / "comment.md"
    assert main(["markdown", str(gate_json), "--out", str(md), "--title", "v3 vs v1"]) == 0
    assert "Blocked: 4 gated metrics regressed" in md.read_text(encoding="utf-8")


def test_markdown_refuses_a_malformed_gate_file(tmp_path: Path) -> None:
    bad = tmp_path / "gate.json"
    bad.write_text('{"metrics": 1}', encoding="utf-8")
    assert main(["markdown", str(bad), "--out", str(tmp_path / "x.md"), "--title", "t"]) == 2


def test_pr_comment_needs_a_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    body = tmp_path / "c.md"
    body.write_text("hello", encoding="utf-8")
    assert main(["pr-comment", "--repo", "o/r", "--pr", "7", "--body-file", str(body)]) == 2


def test_otel_export_then_ingest_reproduces_the_answers(tmp_path: Path) -> None:
    spans = tmp_path / "spans.jsonl"
    assert main(["otel-export", V1, "--jsonl", str(spans), "--capture-content"]) == 0
    assert (
        main(
            [
                "otel-ingest",
                str(spans),
                "--dataset",
                str(ROOT / "demo" / "eval_set.jsonl"),
                "--corpus-dir",
                str(ROOT / "demo" / "corpus"),
                "--out",
                str(tmp_path / "imported"),
            ]
        )
        == 0
    )
    original = {t.case_id: t for t in load_run_traces(V1)}
    imported = load_run_traces(tmp_path / "imported")
    assert len(imported) == len(original)
    for trace in imported:
        assert trace.final_answer == original[trace.case_id].final_answer
        assert trace.retrieved_doc_ids == original[trace.case_id].retrieved_doc_ids
    results = json.loads((tmp_path / "imported" / "eval_results.json").read_text(encoding="utf-8"))
    committed = json.loads((Path(V1) / "eval_results.json").read_text(encoding="utf-8"))
    for key in ("mean_keyword_recall", "mean_groundedness", "success_rate"):
        assert results["aggregates"][key] == committed["aggregates"][key]


def test_the_console_script_is_installed() -> None:
    exe = shutil.which("aire", path=sysconfig.get_path("scripts"))
    assert exe is not None, "the aire console script is not installed; pip install -e ."
    out = subprocess.run([exe, "--help"], capture_output=True, text=True, check=False)
    assert out.returncode == 0
    assert "otel-export" in out.stdout
