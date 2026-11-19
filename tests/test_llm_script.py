"""scripts/run_llm_eval.py --rescore re-scores persisted traces with no API call."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

from aire.llm.deterministic import ExtractiveRAG
from aire.runner.runner import ExperimentRunner, RunnerConfig

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
    assert summary["aggregates"]["n_cases"] == 26
    assert (run_dir / "eval_results.json").exists()
    comparison = json.loads((results_dir / "llm_vs_deterministic.json").read_text("utf-8"))
    assert comparison["summary"]["labels"]["baseline"].startswith("deterministic v2")
    assert (reports_dir / "llm_vs_deterministic_v2.html").exists()
