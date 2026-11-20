"""scripts/run_llm_eval.py --rescore re-scores persisted traces with no API call."""

import asyncio
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from aire.llm.deterministic import ExtractiveRAG
from aire.runner.dataset import load_eval_dataset
from aire.runner.runner import ExperimentRunner, RunnerConfig
from aire.testing.fake_server import FakeLLMServer

ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_llm_eval", ROOT / "scripts" / "run_llm_eval.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rescore_rebuilds_results_and_comparison_from_persisted_traces(tmp_path: Path) -> None:
    script = _load_script()
    corpus = script.load_corpus(ROOT / "demo" / "corpus")
    # Stand-in for a persisted live run: same dataset, a deterministic system.
    runner = ExperimentRunner(ExtractiveRAG("stand_in", corpus, top_k=3), RunnerConfig())
    run_dir = runner.run(
        ROOT / "demo" / "eval_set.jsonl",
        runs_root=tmp_path / "runs",
        corpus=corpus,
        path_root=ROOT,
    )
    results_dir = tmp_path / "results"
    shutil.copytree(ROOT / "results" / "v2_improved", results_dir / "v2_improved")
    reports_dir = tmp_path / "reports"

    script.main(
        [
            "--rescore",
            str(run_dir),
            "--results-dir",
            str(results_dir),
            "--reports-dir",
            str(reports_dir),
        ]
    )

    summary = json.loads((results_dir / "llm_eval_results.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "llm"
    assert summary["aggregates"]["n_cases"] == len(
        load_eval_dataset(ROOT / "demo" / "eval_set.jsonl")
    )
    assert (run_dir / "eval_results.json").exists()
    comparison = json.loads((results_dir / "llm_vs_deterministic.json").read_text("utf-8"))
    assert comparison["summary"]["labels"]["baseline"].startswith("deterministic v2")
    assert (reports_dir / "llm_vs_deterministic_v2.html").exists()


def test_dry_run_states_calls_and_tokens_without_a_key_or_network(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    script.main(
        [
            "--dry-run",
            "--provider",
            "gemini",
            "--model",
            "some-model",
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    out = capsys.readouterr().out
    n_cases = len(load_eval_dataset(ROOT / "demo" / "eval_set.jsonl"))
    assert "provider: gemini" in out
    assert "model: some-model" in out
    assert f"calls: {n_cases}" in out
    assert "estimated tokens:" in out


def test_live_path_runs_end_to_end_against_a_local_server(tmp_path: Path) -> None:
    script = _load_script()
    results_dir = tmp_path / "results"
    shutil.copytree(ROOT / "results" / "v2_improved", results_dir / "v2_improved")

    async def main() -> int:
        async with FakeLLMServer(requests_per_minute=1200) as server:
            await script.amain(
                [
                    "--provider",
                    "local",
                    "--base-url",
                    server.base_url,
                    "--model",
                    "fake-extractive",
                    "--results-dir",
                    str(results_dir),
                    "--reports-dir",
                    str(tmp_path / "reports"),
                    "--runs-dir",
                    str(tmp_path / "runs"),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--rpm",
                    "6000",
                    "--concurrency",
                    "8",
                ]
            )
            return server.served

    served = asyncio.run(main())
    n_cases = len(load_eval_dataset(ROOT / "demo" / "eval_set.jsonl"))
    assert served == n_cases
    persisted = results_dir / "llm_fake-extractive"
    assert (persisted / "traces.jsonl").is_file() and (persisted / "manifest.json").is_file()
    summary = json.loads((results_dir / "llm_eval_results.json").read_text(encoding="utf-8"))
    assert summary["aggregates"]["mean_errors"] == 0.0
    assert summary["total_api_calls"] == n_cases
