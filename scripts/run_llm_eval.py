"""Run the evaluation suite with a real LLM, or re-score a persisted run.

Outputs (relative to the repository root):
    results/llm_<model>/{manifest.json,traces.jsonl,eval_results.json}
                                             the persisted run, replayable
    results/llm_eval_results.json            per-case results + aggregates + API totals
    results/llm_vs_deterministic.json        comparison against results/v2_improved
    reports/llm_vs_deterministic_v2.html     the same comparison as static HTML

Usage (keys come from the environment and are never written anywhere):
    python scripts/run_llm_eval.py --dry-run --provider gemini --model MODEL
    GEMINI_API_KEY=... python scripts/run_llm_eval.py --provider gemini --model MODEL
    python scripts/run_llm_eval.py --provider local --base-url http://localhost:11434/v1 --model M
    python scripts/run_llm_eval.py --rescore [RUN_DIR]

--dry-run prints the provider, the model, the number of calls the run would
make (cache hits excluded) and an upper-bound token estimate, and exits
without a key or any network access. Use it before every live run.

--rescore evaluates the traces of a persisted run with the current metric
code and rewrites the outputs above; it makes no API call. The baseline is
always the committed results/v2_improved.

Retrieval is AIRE's local IDF-weighted retriever; only generation is remote.
Generation goes through a ModelClient: a shared adaptive rate limiter
(--rpm, --tpm), retries with the server's requested wait or jittered backoff,
a quota circuit, and a response cache (--cache-dir), so an interrupted run
can be resumed without paying twice. Waits are recorded on each span and
excluded from latency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aire.evaluator.engine import EvaluationEngine, write_eval_results
from aire.llm.deterministic import KeywordRetriever
from aire.providers.base import ChatModel
from aire.providers.cache import ResponseCache, request_key
from aire.providers.client import ModelClient, RetryPolicy
from aire.providers.local import LocalOpenAICompatModel
from aire.providers.ratelimit import AdaptiveRateLimiter
from aire.providers.types import ChatRequest, Message
from aire.regression import RegressionThresholds, compare_runs, write_regression_report
from aire.report import render_comparison_report
from aire.runner.async_runner import AsyncExperimentRunner, AsyncRunnerConfig
from aire.runner.dataset import load_eval_dataset
from aire.systems.prompts import RAG_SYSTEM_PROMPT, rag_user_prompt
from aire.systems.rag import LLMRagSystem
from aire.tracing.storage import atomic_write_text, load_run_traces

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "demo" / "eval_set.jsonl"
MODEL = "gemini-2.5-flash"  # the model of the committed historical run
PROVIDERS = ("gemini", "openai", "anthropic", "local")
TOP_K = 3
MAX_OUTPUT_TOKENS = 256
RUN_FILES = ("manifest.json", "traces.jsonl")


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.md"))}


def persisted_run_dir(results_dir: Path, model: str = MODEL) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in model)
    return results_dir / f"llm_{safe}"


def provider_name(provider: str) -> str:
    """The gen_ai.provider.name each adapter reports (the cache keys on it)."""
    return {"gemini": "gcp.gemini", "openai": "openai", "anthropic": "anthropic"}.get(
        provider, "ollama"
    )


def sampling_temperature(provider: str) -> float | None:
    """0.0 where the provider accepts it; anthropic 2.x removed the parameter."""
    return None if provider == "anthropic" else 0.0


def build_chat_model(provider: str, base_url: str | None) -> ChatModel:
    """The adapter for --provider; SDK adapters import their SDK only here."""
    if provider == "gemini":
        from aire.providers.gemini_sdk import GeminiModel

        return GeminiModel(api_key_env="GEMINI_API_KEY")
    if provider == "openai":
        from aire.providers.openai_sdk import OpenAIModel

        return OpenAIModel(base_url=base_url)
    if provider == "anthropic":
        from aire.providers.anthropic_sdk import AnthropicModel

        return AnthropicModel()
    return LocalOpenAICompatModel(base_url or "http://localhost:11434/v1", provider="ollama")


def planned_requests(corpus: dict[str, str], provider: str, model: str) -> list[ChatRequest]:
    """The exact requests a live run sends (built the same way as LLMRagSystem)."""
    retriever = KeywordRetriever(corpus, top_k=TOP_K, weighting="idf")
    requests = []
    for case in load_eval_dataset(DATASET):
        docs = [(r.doc_id, r.text) for r in retriever.retrieve(case.input)]
        requests.append(
            ChatRequest(
                model=model,
                messages=(
                    Message("system", RAG_SYSTEM_PROMPT),
                    Message("user", rag_user_prompt(case.input, docs)),
                ),
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=sampling_temperature(provider),
            )
        )
    return requests


def dry_run(corpus: dict[str, str], args: argparse.Namespace) -> None:
    requests = planned_requests(corpus, args.provider, args.model)
    cache = ResponseCache(args.cache_dir)
    key_provider = provider_name(args.provider)
    pending = [r for r in requests if cache.get(request_key(key_provider, r)) is None]
    tokens = sum(r.estimated_tokens() for r in pending)
    print(f"provider: {args.provider}")
    print(f"model: {args.model}")
    print(f"cases: {len(requests)} (already cached: {len(requests) - len(pending)})")
    print(f"calls: {len(pending)} (plus retries if the provider rate-limits)")
    print(
        f"estimated tokens: at most {tokens} "
        f"(prompt characters / 4 + {MAX_OUTPUT_TOKENS} output tokens per call)"
    )
    print("no request was sent")


async def live_run(corpus: dict[str, str], args: argparse.Namespace) -> Path:
    """Run the suite against the live model and persist manifest + traces."""
    client = ModelClient(
        build_chat_model(args.provider, args.base_url),
        limiter=AdaptiveRateLimiter(requests_per_minute=args.rpm, tokens_per_minute=args.tpm),
        cache=ResponseCache(args.cache_dir),
        retry=RetryPolicy(max_attempts=5, base_delay_s=1.0, max_delay_s=60.0),
        rng=random.Random(0),
    )
    sut = LLMRagSystem(
        f"llm_rag_{args.model}",
        corpus,
        client,
        model=args.model,
        top_k=TOP_K,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=sampling_temperature(args.provider),
    )
    runner = AsyncExperimentRunner(
        sut, AsyncRunnerConfig(concurrency=args.concurrency, timeout_s=600.0)
    )
    outcome = await runner.arun(
        DATASET, runs_root=args.runs_dir, run_name=sut.name, corpus=corpus, path_root=ROOT
    )
    print(
        f"live run: {outcome.n_cases} cases, {outcome.n_failed} failed, "
        f"{outcome.wall_time_s:.1f}s, client {client.stats()}"
    )
    target = persisted_run_dir(args.results_dir, args.model)
    target.mkdir(parents=True, exist_ok=True)
    for fname in RUN_FILES:
        shutil.copy2(outcome.run_dir / fname, target / fname)
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
    # Totals come from the recorded traces, so they count exactly what was stored.
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
            labels=("deterministic v2 (extractive)", f"LLM RAG ({results['generator']})"),
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
    parser.add_argument("--provider", choices=PROVIDERS, default="gemini")
    parser.add_argument("--model", default=MODEL, help=f"model id (default: {MODEL})")
    parser.add_argument("--base-url", default=None, help="endpoint for --provider local or openai")
    parser.add_argument("--rpm", type=float, default=10.0, help="requests per minute budget")
    parser.add_argument("--tpm", type=float, default=None, help="tokens per minute budget")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the planned calls and tokens; send nothing"
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
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "llm_cache")
    return parser


async def amain(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    corpus = load_corpus(ROOT / "demo" / "corpus")
    if args.dry_run:
        dry_run(corpus, args)
        return
    if args.rescore is None:
        run_dir = await live_run(corpus, args)
    elif args.rescore:
        run_dir = Path(args.rescore)
    else:
        run_dir = persisted_run_dir(args.results_dir, args.model)
    results = evaluate_and_report(run_dir, corpus, args.results_dir, args.reports_dir)
    agg = results["aggregates"]
    means = {k: v for k, v in agg.items() if k.startswith("mean_") and k != "mean_cost_usd"}
    print(f"LLM RAG ({results['generator']}): {agg['n_success']}/{agg['n_cases']} clean |", means)


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(amain(argv))


if __name__ == "__main__":
    main()
