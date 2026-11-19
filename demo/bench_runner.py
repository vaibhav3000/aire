"""Measure the async runner against a rate-limited local server.

Runs the whole eval set through the served stack (LLMRagSystem, ModelClient,
stdlib HTTP adapter, FakeLLMServer on 127.0.0.1) four ways:

1. sequential: one case at a time,
2. concurrent: 8 workers, client budget equal to the server's limit,
3. over-eager: 8 workers, client told twice the server's limit, so the
   adaptive limiter has to absorb real 429s,
4. warm cache: the concurrent run again over a filled response cache.

Writes results/runner_benchmark.json with the configuration, the machine and
every measured number. Timings vary between machines and runs; the file
records one real run, labelled with its date. Usage:

    python demo/bench_runner.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import platform
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from aire.providers.cache import ResponseCache
from aire.providers.client import ModelClient, RetryPolicy
from aire.providers.local import LocalOpenAICompatModel
from aire.providers.ratelimit import AdaptiveRateLimiter
from aire.runner.async_runner import AsyncExperimentRunner, AsyncRunnerConfig
from aire.runner.dataset import load_eval_dataset
from aire.systems.rag import LLMRagSystem
from aire.testing.fake_server import FakeLLMServer
from aire.tracing.storage import atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "demo" / "eval_set.jsonl"
SERVER_RPM = 3000.0
SERVICE_TIME_S = 0.1
CONCURRENCY = 8


def load_corpus() -> dict[str, str]:
    return {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted((ROOT / "demo" / "corpus").glob("*.md"))
    }


async def measure(
    label: str,
    corpus: dict[str, str],
    runs_root: Path,
    *,
    concurrency: int,
    client_rpm: float,
    cache: ResponseCache | None,
) -> dict[str, Any]:
    async with FakeLLMServer(
        requests_per_minute=SERVER_RPM, service_time_s=SERVICE_TIME_S
    ) as server:
        limiter = AdaptiveRateLimiter(requests_per_minute=client_rpm, tokens_per_minute=None)
        client = ModelClient(
            LocalOpenAICompatModel(server.base_url, timeout_s=30),
            limiter=limiter,
            cache=cache,
            retry=RetryPolicy(max_attempts=8, base_delay_s=0.1, max_delay_s=10.0),
            rng=random.Random(0),
        )
        sut = LLMRagSystem(label, corpus, client, model="fake-extractive", deterministic=True)
        runner = AsyncExperimentRunner(
            sut, AsyncRunnerConfig(concurrency=concurrency, timeout_s=120)
        )
        outcome = await runner.arun(DATASET, runs_root=runs_root, run_name=label)
        return {
            "label": label,
            "concurrency": concurrency,
            "client_requests_per_minute": client_rpm,
            "cache": cache is not None,
            "wall_time_s": round(outcome.wall_time_s, 3),
            "cases_per_s": round(outcome.cases_per_s, 2),
            "n_cases": outcome.n_cases,
            "n_failed": outcome.n_failed,
            "server": {
                "requests": server.requests,
                "served": server.served,
                "rate_limited": server.rate_limited,
            },
            "client": client.stats(),
            "limiter": limiter.stats(),
        }


async def main_async() -> dict[str, Any]:
    corpus = load_corpus()
    work = Path(tempfile.mkdtemp(prefix="aire-bench-"))
    try:
        cache = ResponseCache(work / "cache")
        runs = [
            await measure(
                "sequential",
                corpus,
                work / "runs",
                concurrency=1,
                client_rpm=SERVER_RPM,
                cache=None,
            ),
            await measure(
                "concurrent",
                corpus,
                work / "runs",
                concurrency=CONCURRENCY,
                client_rpm=SERVER_RPM,
                cache=cache,
            ),
            await measure(
                "over_eager",
                corpus,
                work / "runs",
                concurrency=CONCURRENCY,
                client_rpm=2 * SERVER_RPM,
                cache=None,
            ),
            await measure(
                "warm_cache",
                corpus,
                work / "runs",
                concurrency=CONCURRENCY,
                client_rpm=SERVER_RPM,
                cache=cache,
            ),
        ]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    sequential = runs[0]["wall_time_s"]
    return {
        "description": (
            "Served stack against FakeLLMServer on 127.0.0.1: one real run of each "
            "configuration over the whole eval set. Timings depend on the machine."
        ),
        "recorded_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "machine": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "processor": platform.processor() or "unknown",
        },
        "dataset": "demo/eval_set.jsonl",
        "n_cases": len(load_eval_dataset(DATASET)),
        "server": {
            "requests_per_minute": SERVER_RPM,
            "service_time_s": SERVICE_TIME_S,
            "retry_hint": "header",
        },
        "runs": runs,
        "speedup_vs_sequential": {
            r["label"]: round(sequential / r["wall_time_s"], 2)
            for r in runs
            if r["wall_time_s"] > 0
        },
    }


def main() -> None:
    report = asyncio.run(main_async())
    out = ROOT / "results" / "runner_benchmark.json"
    atomic_write_text(out, json.dumps(report, indent=2) + "\n")
    for run in report["runs"]:
        print(
            f"{run['label']:<11} wall={run['wall_time_s']:>7.3f}s  {run['cases_per_s']:>7.2f} cases/s  "
            f"failed={run['n_failed']}  server requests={run['server']['requests']} "
            f"429s={run['server']['rate_limited']}"
        )
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
