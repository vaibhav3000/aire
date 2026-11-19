"""Run the 26-case evaluation suite with a REAL LLM responder, or re-score a
persisted run without calling the model.

Outputs (relative to the repository root):
    results/llm_<model>/{manifest.json,traces.jsonl,eval_results.json}
                                             the persisted run, replayable
    results/llm_eval_results.json            per-case results + aggregates + API totals
    results/llm_vs_deterministic.json        comparison against results/v2_improved
    reports/llm_vs_deterministic_v2.html     the same comparison as static HTML

Usage (the key comes from the environment and is never written anywhere):
    GEMINI_API_KEY=... python scripts/run_llm_eval.py
    python scripts/run_llm_eval.py --rescore [RUN_DIR]

--rescore evaluates the traces of a persisted run (default
results/llm_<model>/) with the current metric code and rewrites the outputs
above; it makes no API call. The baseline is always the committed
results/v2_improved, so the comparison is replayable from the repository.

The retrieval step is AIRE's local IDF-weighted retriever; only generation is
remote. Calls are paced to stay under the provider's rate limit. The pacing
runs before the runner's timer starts, and the provider's retry waits are
recorded on each span and excluded from latency.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aire.evaluator.engine import EvaluationEngine, write_eval_results
from aire.llm.openai_compat import OpenAIChatResponder
from aire.llm.provider import OpenAICompatProvider
from aire.regression import RegressionThresholds, compare_runs, write_regression_report
from aire.report import render_comparison_report
from aire.runner import ExperimentRunner, RunnerConfig
from aire.runner.dataset import load_eval_dataset
from aire.tracing.storage import atomic_write_text, load_run_traces

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "demo" / "eval_set.jsonl"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
MODEL = "gemini-2.5-flash"
RUN_FILES = ("manifest.json", "traces.jsonl")


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.md"))}


def persisted_run_dir(results_dir: Path) -> Path:
    return results_dir / f"llm_{MODEL}"


def live_run(corpus: dict[str, str], results_dir: Path) -> Path:
    """Run the suite against the live model and persist manifest + traces."""
    provider = OpenAICompatProvider(
        base_url=GEMINI_BASE_URL,
        api_key_env="GEMINI_API_KEY",
        model=MODEL,
        temperature=0.0,
        timeout_s=60.0,
        max_retries=3,
        max_backoff_s=30.0,
    )
    sut = OpenAIChatResponder(
        f"llm_rag_{MODEL}",
        corpus,
        provider,
        top_k=3,
        weighting="idf",
        min_call_interval_s=5.0,  # stay under 15 requests/minute
    )
    # The runner's timeout must outlast the provider's worst case, or a
    # timed-out worker thread would keep calling the API behind the runner.
    runner = ExperimentRunner(sut, RunnerConfig(timeout_s=provider.max_call_duration_s + 10.0))
    run_dir = runner.run(
        DATASET, runs_root=ROOT / "runs", run_name=sut.name, corpus=corpus, path_root=ROOT
    )
    target = persisted_run_dir(results_dir)
    target.mkdir(parents=True, exist_ok=True)
    for fname in RUN_FILES:
        shutil.copy2(run_dir / fname, target / fname)
    return target


def evaluate_and_report(
    run_dir: Path, corpus: dict[str, str], results_dir: Path, reports_dir: Path
) -> dict[str, Any]:
    """Score a persisted run with the current metric code and write all outputs."""
    for fname in RUN_FILES:
        if not (run_dir / fname).exists():
            raise SystemExit(f"{run_dir / fname} not found; nothing to score")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    traces = load_run_traces(run_dir)
    results = EvaluationEngine(corpus=corpus).evaluate_run(traces, load_eval_dataset(DATASET))
    results["mode"] = "llm"
    results["generator"] = manifest.get("system_config", {}).get("generator")
    # Totals come from the recorded traces, so a timed-out worker thread that
    # finishes later cannot inflate them.
    api_usage = [t.usage for t in traces if t.usage.token_source == "api"]
    results["total_api_prompt_tokens"] = sum(u.prompt_tokens for u in api_usage)
    results["total_api_completion_tokens"] = sum(u.completion_tokens for u in api_usage)
    results["total_api_calls"] = len(api_usage)
    write_eval_results(run_dir, results)
    atomic_write_text(results_dir / "llm_eval_results.json", json.dumps(results, indent=2))

    baseline_dir = results_dir / "v2_improved"
    if (baseline_dir / "eval_results.json").exists():
        baseline = json.loads((baseline_dir / "eval_results.json").read_text(encoding="utf-8"))
        comparison = compare_runs(
            baseline_dir,
            run_dir,
            thresholds=RegressionThresholds(default=0.02, overrides={"latency_ms": 0.5}),
            labels=("deterministic v2 (extractive)", f"LLM RAG ({MODEL})"),
        )
        write_regression_report(comparison, results_dir / "llm_vs_deterministic.json")
        render_comparison_report(
            comparison,
            baseline,
            results,
            reports_dir / "llm_vs_deterministic_v2.html",
            title="deterministic v2 vs real-LLM RAG",
        )
        print("comparison:", comparison["summary"])
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the evaluation suite with a real LLM, or re-score a persisted run."
    )
    parser.add_argument(
        "--rescore",
        nargs="?",
        const="",
        metavar="RUN_DIR",
        help="re-score a persisted run (default results/llm_<model>/) without calling the API",
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    corpus = load_corpus(ROOT / "demo" / "corpus")
    if args.rescore is None:
        run_dir = live_run(corpus, args.results_dir)
    else:
        run_dir = Path(args.rescore) if args.rescore else persisted_run_dir(args.results_dir)
    results = evaluate_and_report(run_dir, corpus, args.results_dir, args.reports_dir)
    agg = results["aggregates"]
    means = {k: v for k, v in agg.items() if k.startswith("mean_") and k != "mean_cost_usd"}
    print(f"LLM RAG ({MODEL}): {agg['n_success']}/{agg['n_cases']} cases clean |", means)


if __name__ == "__main__":
    main()
