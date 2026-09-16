"""Run the AIRE demo: three RAG versions over one evaluation set, with
regression reports and static HTML dashboards.

Usage (from the aire/ directory):
    python demo/run_demo.py

Produces:
    runs/          stored traces per version (manifest + traces.jsonl + eval_results.json)
    reports/demo_vB_vs_vA.html   improvement report
    reports/demo_vC_vs_vA.html   deliberate-regression report (groundedness drop)
    results/       committed copies of the JSON reports

The demo is fully offline: the systems under test are deterministic extractive
RAG variants (src/aire/llm/deterministic.py). No API keys, no network.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aire.evaluator.engine import EngineConfig, EvaluationEngine, write_eval_results  # noqa: E402
from aire.llm.deterministic import ExtractiveRAG  # noqa: E402
from aire.regression import RegressionThresholds, compare_runs, write_regression_report  # noqa: E402
from aire.report import render_comparison_report  # noqa: E402
from aire.runner import ExperimentRunner, RunnerConfig  # noqa: E402
from aire.tracing.storage import load_run_traces  # noqa: E402


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    return {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted(corpus_dir.glob("*.md"))
    }


def evaluate_version(
    sut: ExtractiveRAG,
    dataset_path: Path,
    corpus: dict[str, str],
    runs_root: Path,
) -> Path:
    runner = ExperimentRunner(sut, RunnerConfig(timeout_s=10.0))
    run_dir = runner.run(dataset_path, runs_root=runs_root, run_name=sut.name)
    traces = load_run_traces(run_dir)
    engine = EvaluationEngine(corpus=corpus)
    from aire.runner.dataset import load_eval_dataset

    cases = load_eval_dataset(dataset_path)
    results = engine.evaluate_run(traces, cases)
    write_eval_results(run_dir, results)
    return run_dir


def main() -> None:
    dataset = ROOT / "demo" / "eval_set.jsonl"
    corpus_dir = ROOT / "demo" / "corpus"
    if not dataset.exists() or not corpus_dir.exists():
        raise SystemExit("demo assets missing: expected demo/eval_set.jsonl and demo/corpus/")
    corpus = load_corpus(corpus_dir)
    runs_root = ROOT / "runs"
    runs_root.mkdir(exist_ok=True)

    versions = {
        "v1_baseline": ExtractiveRAG("v1_baseline", corpus, top_k=1, min_score=0.05),
        "v2_improved": ExtractiveRAG("v2_improved", corpus, top_k=3, min_score=0.05, weighting="idf"),
        "v3_fluent_regression": ExtractiveRAG(
            "v3_fluent_regression", corpus, top_k=3, min_score=0.05, style="fluent"
        ),
    }

    run_dirs: dict[str, Path] = {}
    for name, sut in versions.items():
        print(f"running {name} ...", flush=True)
        run_dirs[name] = evaluate_version(sut, dataset, corpus, runs_root)

    results_dir = ROOT / "results"
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    comparisons = [
        ("v2_improved", "v1_baseline", "v2_improved_vs_v1_baseline"),
        ("v3_fluent_regression", "v1_baseline", "v3_fluent_regression_vs_v1_baseline"),
    ]
    for cand, base, stem in comparisons:
        # Latency differences below 10% are noise at sub-millisecond scale;
        # the override exists precisely so teams can encode that judgment.
        thresholds = RegressionThresholds(default=0.02, overrides={"latency_ms": 0.10})
        comparison = compare_runs(run_dirs[base], run_dirs[cand], thresholds=thresholds)
        write_regression_report(comparison, results_dir / f"{stem}.json")
        base_results = json.loads((run_dirs[base] / "eval_results.json").read_text(encoding="utf-8"))
        cand_results = json.loads((run_dirs[cand] / "eval_results.json").read_text(encoding="utf-8"))
        render_comparison_report(
            comparison,
            base_results,
            cand_results,
            reports_dir / f"{stem}.html",
            title=stem.replace("_", " "),
        )
        summary = comparison["summary"]
        print(f"[{stem}] {summary['verdict']}: "
              f"improved={summary['n_improved']} regressed={summary['n_regressed']} "
              f"neutral={summary['n_neutral']}", flush=True)

    # Commit copies of run JSON artifacts into results/ for reproducibility.
    if results_dir.exists():
        for run_name, run_dir in run_dirs.items():
            target = results_dir / run_name
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            for fname in ("manifest.json", "traces.jsonl", "eval_results.json"):
                src = run_dir / fname
                if src.exists():
                    shutil.copy2(src, target / fname)
    print("demo complete.")


if __name__ == "__main__":
    main()
