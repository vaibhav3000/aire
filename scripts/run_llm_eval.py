"""Run the 26-case evaluation suite with a REAL LLM responder.

Real-LLM mode is recorded separately from the deterministic demo:
    results/llm_eval_results.json            per-case + aggregates
    results/llm_vs_deterministic.json        regression-engine comparison
    reports/llm_vs_deterministic_v2.html     static report

Usage (key comes from the environment; never committed):
    GEMINI_API_KEY=... python scripts/run_llm_eval.py

The retrieval step is AIRE's local IDF-weighted retriever; only generation is
remote. Calls are rate-limited to stay under the provider's quota; the pacing
sleep happens outside the timed region so latency metrics measure the system.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aire.evaluator.engine import EvaluationEngine, write_eval_results  # noqa: E402
from aire.llm.openai_compat import OpenAIChatResponder  # noqa: E402
from aire.llm.provider import OpenAICompatProvider  # noqa: E402
from aire.regression import RegressionThresholds, compare_runs, write_regression_report  # noqa: E402
from aire.report import render_comparison_report  # noqa: E402
from aire.runner import ExperimentRunner, RunnerConfig  # noqa: E402
from aire.runner.dataset import load_eval_dataset  # noqa: E402
from aire.tracing.storage import load_run_traces  # noqa: E402

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.md"))}


def main() -> None:
    dataset = ROOT / "demo" / "eval_set.jsonl"
    corpus = load_corpus(ROOT / "demo" / "corpus")
    model = "gemini-2.5-flash"
    provider = OpenAICompatProvider(
        base_url=GEMINI_BASE_URL,
        api_key_env="GEMINI_API_KEY",
        model=model,
        temperature=0.0,
    )
    sut = OpenAIChatResponder(
        f"llm_rag_{model}", corpus, provider, top_k=3, weighting="idf",
        min_call_interval_s=5.0,  # stay under 15 requests/minute
    )
    runner = ExperimentRunner(sut, RunnerConfig(timeout_s=150.0))
    run_dir = runner.run(dataset, runs_root=ROOT / "runs", run_name=sut.name)

    engine = EvaluationEngine(corpus=corpus)
    results = engine.evaluate_run(load_run_traces(run_dir), load_eval_dataset(dataset))
    results["mode"] = "llm"
    results["generator"] = sut.config["generator"]
    results["total_api_prompt_tokens"] = sut.prompt_tokens
    results["total_api_completion_tokens"] = sut.completion_tokens
    results["total_api_calls"] = sut.calls
    write_eval_results(run_dir, results)

    # Commit a copy + a comparison against the deterministic improved version.
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "llm_eval_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    det_run_dir = None
    for d in sorted((ROOT / "runs").glob("run_*")):
        m = d / "manifest.json"
        if m.exists() and json.loads(m.read_text(encoding="utf-8")).get("name") == "v2_improved":
            det_run_dir = d
    if det_run_dir is not None:
        det_results = json.loads((det_run_dir / "eval_results.json").read_text(encoding="utf-8"))
        comparison = compare_runs(
            det_run_dir, run_dir,
            thresholds=RegressionThresholds(default=0.02, overrides={"latency_ms": 0.5}),
            labels=("deterministic v2 (extractive)", f"LLM RAG ({model})"),
        )
        write_regression_report(comparison, results_dir / "llm_vs_deterministic.json")
        render_comparison_report(
            comparison, det_results, results,
            ROOT / "reports" / "llm_vs_deterministic_v2.html",
            title="deterministic v2 vs real-LLM RAG",
        )
        print("comparison:", comparison["summary"])

    agg = results["aggregates"]
    print(f"LLM RAG ({model}): {agg['n_success']}/{agg['n_cases']} cases clean |",
          {k: v for k, v in agg.items() if k.startswith("mean_") and k != "mean_cost_usd"})


if __name__ == "__main__":
    main()
