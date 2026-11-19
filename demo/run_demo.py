"""Run the AIRE demo: three RAG versions over one evaluation set, with
regression reports, static HTML dashboards and a retrieval ablation.

Usage (from the repository root, after `pip install -e .`):
    python demo/run_demo.py

Produces:
    runs/<run_id>/                     raw run directories (gitignored)
    results/<version>/                 committed manifest, traces and eval results
    results/<cand>_vs_<base>.json      regression comparisons
    reports/<cand>_vs_<base>.html      the same comparisons as static HTML
    results/retrieval_ablation.json    overlap vs idf retrieval at K=1 and K=3

The committed copies under results/ are replaced only when something other
than timing changed (ids, timestamps and latencies differ on every run), so
re-running the demo on an unchanged tree leaves git clean. Sub-millisecond
latencies of this offline demo are noise, so latency is reported in the
comparisons but excluded from their verdict counts.

The demo is fully offline: the systems under test are deterministic extractive
RAG variants (src/aire/llm/deterministic.py). No API keys, no network.
"""

from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path
from typing import Any

from aire.evaluator import retrieval_metrics as R
from aire.evaluator.engine import EvaluationEngine, write_eval_results
from aire.llm.deterministic import ExtractiveRAG, KeywordRetriever
from aire.regression import RegressionThresholds, compare_runs, write_regression_report
from aire.report import render_comparison_report
from aire.runner import ExperimentRunner, RunnerConfig
from aire.runner.dataset import EvalCase, load_eval_dataset
from aire.tracing.storage import atomic_write_text, load_run_traces

ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = ("manifest.json", "traces.jsonl", "eval_results.json")
# Fields that change on every run without the system changing.
VOLATILE_KEYS = frozenset(
    {
        "run_id",
        "trace_id",
        "created_at",
        "started_at",
        "latency_ms",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
    }
)
COMPARISONS = (
    ("v2_improved", "v1_baseline"),
    ("v3_fluent_regression", "v1_baseline"),
)


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.md"))}


def evaluate_version(
    sut: ExtractiveRAG,
    dataset_path: Path,
    corpus: dict[str, str],
    runs_root: Path,
) -> Path:
    runner = ExperimentRunner(sut, RunnerConfig(timeout_s=10.0))
    run_dir = runner.run(
        dataset_path, runs_root=runs_root, run_name=sut.name, corpus=corpus, path_root=ROOT
    )
    results = EvaluationEngine(corpus=corpus).evaluate_run(
        load_run_traces(run_dir), load_eval_dataset(dataset_path)
    )
    write_eval_results(run_dir, results)
    return run_dir


def _without_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_volatile(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_without_volatile(v) for v in value]
    return value


def _stable_content(path: Path) -> Any:
    """A file's JSON content with run-specific fields removed (None if absent)."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [_without_volatile(json.loads(line)) for line in text.splitlines() if line.strip()]
    return _without_volatile(json.loads(text))


def publish_run(run_dir: Path, target: Path) -> bool:
    """Copy a fresh run into results/ unless only timing fields differ.

    Returns True when the committed copy was replaced.
    """
    if all(_stable_content(run_dir / f) == _stable_content(target / f) for f in RUN_FILES):
        return False
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for fname in RUN_FILES:
        shutil.copy2(run_dir / fname, target / fname)
    return True


def retrieval_ablation(corpus: dict[str, str], cases: list[EvalCase]) -> dict[str, Any]:
    """Overlap vs idf weighting at fixed cut-offs, over the answerable cases.

    Separates the effect of the weighting from the effect of retrieving more
    documents, which the v1 (overlap, top 1) vs v2 (idf, top 3) comparison
    mixes together.
    """
    answerable = [c for c in cases if c.supporting_doc_ids]
    table: dict[str, dict[str, float]] = {}
    for weighting in ("overlap", "idf"):
        retriever = KeywordRetriever(corpus, top_k=3, weighting=weighting)
        ranked = [[r.doc_id for r in retriever.retrieve(c.input)] for c in answerable]
        gold = [c.supporting_doc_ids for c in answerable]
        table[weighting] = {
            "recall_at_1": round(
                statistics.fmean(
                    R.retrieval_recall_at_k(r, g, 1) for r, g in zip(ranked, gold, strict=True)
                ),
                4,
            ),
            "recall_at_3": round(
                statistics.fmean(
                    R.retrieval_recall_at_k(r, g, 3) for r, g in zip(ranked, gold, strict=True)
                ),
                4,
            ),
            "mrr_at_3": round(
                statistics.fmean(
                    R.retrieval_mrr(r, g, 3) for r, g in zip(ranked, gold, strict=True)
                ),
                4,
            ),
        }
    return {
        "description": (
            "KeywordRetriever with top_k=3 over the answerable cases; recall and MRR "
            "at fixed cut-offs, so weighting and K are compared separately"
        ),
        "n_cases": len(answerable),
        "retrievers": table,
    }


def main() -> None:
    dataset = ROOT / "demo" / "eval_set.jsonl"
    corpus_dir = ROOT / "demo" / "corpus"
    if not dataset.exists() or not corpus_dir.exists():
        raise SystemExit("demo assets missing: expected demo/eval_set.jsonl and demo/corpus/")
    corpus = load_corpus(corpus_dir)
    runs_root = ROOT / "runs"
    results_dir = ROOT / "results"
    reports_dir = ROOT / "reports"

    versions = {
        "v1_baseline": ExtractiveRAG("v1_baseline", corpus, top_k=1, min_score=0.05),
        "v2_improved": ExtractiveRAG(
            "v2_improved", corpus, top_k=3, min_score=0.05, weighting="idf"
        ),
        # Same retrieval as v2 (idf); the only change is the fluent style, so the
        # regression the evaluator catches is attributable to the style change.
        "v3_fluent_regression": ExtractiveRAG(
            "v3_fluent_regression", corpus, top_k=3, min_score=0.05, weighting="idf", style="fluent"
        ),
    }

    for name, sut in versions.items():
        run_dir = evaluate_version(sut, dataset, corpus, runs_root)
        replaced = publish_run(run_dir, results_dir / name)
        status = "updated results/" if replaced else "unchanged apart from timing:"
        print(f"{name}: {status} {name}", flush=True)

    for cand, base in COMPARISONS:
        stem = f"{cand}_vs_{base}"
        comparison = compare_runs(
            results_dir / base,
            results_dir / cand,
            thresholds=RegressionThresholds(default=0.02),
            labels=(base, cand),
            ignore={"latency_ms"},
        )
        write_regression_report(comparison, results_dir / f"{stem}.json")
        base_results = json.loads((results_dir / base / "eval_results.json").read_text("utf-8"))
        cand_results = json.loads((results_dir / cand / "eval_results.json").read_text("utf-8"))
        render_comparison_report(
            comparison,
            base_results,
            cand_results,
            reports_dir / f"{stem}.html",
            title=stem.replace("_", " "),
        )
        summary = comparison["summary"]
        print(
            f"[{stem}] {summary['verdict']}: improved={summary['n_improved']} "
            f"regressed={summary['n_regressed']} neutral={summary['n_neutral']} "
            f"(latency reported, not counted)",
            flush=True,
        )

    ablation = retrieval_ablation(corpus, load_eval_dataset(dataset))
    atomic_write_text(results_dir / "retrieval_ablation.json", json.dumps(ablation, indent=2))
    for weighting, row in ablation["retrievers"].items():
        print(f"[retrieval ablation] {weighting}: {row}", flush=True)
    print("demo complete.")


if __name__ == "__main__":
    main()
