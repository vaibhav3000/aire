"""The aire command.

    aire gate BASELINE CANDIDATE [aire-gate options]   the CI regression gate
    aire report BASELINE CANDIDATE --out diff.html     the HTML diff report
    aire markdown GATE_JSON --out comment.md           the PR comment body
    aire pr-comment --repo O/R --pr N --body-file F    create or update the sticky comment
    aire otel-export RUN_DIR --jsonl F | --endpoint U  traces as OpenTelemetry GenAI spans
    aire otel-ingest SPANS --dataset D --out RUN_DIR   OpenTelemetry traces as an aire run

Exit codes follow aire-gate: 0 success, 1 regression (gate only), 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .ci.github import GitHubAPI, GitHubError, HeadMismatch, upsert_sticky_comment
from .evaluator.engine import EvaluationEngine, write_eval_results
from .regression.compare import compare_runs
from .regression.gate import run_gate
from .report.html import render_comparison_report
from .report.markdown import render_gate_markdown
from .runner.dataset import load_eval_dataset
from .telemetry.otlp_json import read_jsonl, spans_to_traces, write_jsonl
from .tracing.storage import RunStore, atomic_write_text, load_run_traces

EXIT_OK, EXIT_USAGE = 0, 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aire", description="AI Reliability and Evaluation Engine"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser("gate", help="fail when a candidate run regresses (see aire-gate --help)")
    gate.add_argument("args", nargs=argparse.REMAINDER)

    report = sub.add_parser("report", help="render the HTML diff report for two evaluated runs")
    report.add_argument("baseline", type=Path)
    report.add_argument("candidate", type=Path)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--title", default=None)
    report.add_argument("--ignore", action="append", default=[], metavar="METRIC")

    markdown = sub.add_parser("markdown", help="render a gate report (JSON) as the PR comment")
    markdown.add_argument("gate_json", type=Path)
    markdown.add_argument("--out", type=Path, required=True)
    markdown.add_argument("--title", required=True)
    markdown.add_argument("--details-url", default=None)

    comment = sub.add_parser("pr-comment", help="create or update the sticky PR comment")
    comment.add_argument("--repo", required=True, help="owner/name")
    comment.add_argument("--pr", type=int, required=True)
    comment.add_argument("--body-file", type=Path, required=True)
    comment.add_argument("--head-sha", default=None, help="only post if the PR head is this commit")
    comment.add_argument("--api-url", default="https://api.github.com")

    export = sub.add_parser("otel-export", help="export a run's traces as OpenTelemetry spans")
    export.add_argument("run_dir", type=Path)
    target = export.add_mutually_exclusive_group(required=True)
    target.add_argument("--jsonl", type=Path, help="write OTLP JSON lines (no extra needed)")
    target.add_argument("--endpoint", help="OTLP/HTTP traces endpoint (needs aire[otel])")
    export.add_argument(
        "--capture-content", action="store_true", help="include answers and tool arguments"
    )
    export.add_argument("--service-name", default="aire")

    ingest = sub.add_parser("otel-ingest", help="turn OTLP JSON lines into an evaluated aire run")
    ingest.add_argument("spans", type=Path)
    ingest.add_argument("--dataset", type=Path, required=True)
    ingest.add_argument("--out", type=Path, required=True, help="run directory to create")
    ingest.add_argument("--corpus-dir", type=Path, default=None)
    ingest.add_argument("--run-id", default=None)
    return parser


def _report(args: argparse.Namespace) -> int:
    comparison = compare_runs(args.baseline, args.candidate, ignore=set(args.ignore))
    base = json.loads((args.baseline / "eval_results.json").read_text(encoding="utf-8"))
    cand = json.loads((args.candidate / "eval_results.json").read_text(encoding="utf-8"))
    title = args.title or f"{args.candidate.name} vs {args.baseline.name}"
    render_comparison_report(comparison, base, cand, args.out, title=title)
    print(f"wrote {args.out}")
    return EXIT_OK


def _markdown(args: argparse.Namespace) -> int:
    report = json.loads(args.gate_json.read_text(encoding="utf-8"))
    body = render_gate_markdown(report, title=args.title, details_url=args.details_url)
    atomic_write_text(args.out, body + "\n")
    print(f"wrote {args.out} ({len(body)} characters)")
    return EXIT_OK


def _pr_comment(args: argparse.Namespace) -> int:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("aire pr-comment: GITHUB_TOKEN is not set", file=sys.stderr)
        return EXIT_USAGE
    body = args.body_file.read_text(encoding="utf-8")
    api = GitHubAPI(token, api_url=args.api_url)
    try:
        outcome = upsert_sticky_comment(
            api, args.repo, args.pr, body, expected_head_sha=args.head_sha
        )
    except HeadMismatch as exc:
        # The PR moved on; its newer commit gets its own evaluation and comment.
        print(f"aire pr-comment: not posting a stale result: {exc}")
        return EXIT_OK
    print(f"aire pr-comment: {outcome} the comment on {args.repo}#{args.pr}")
    return EXIT_OK


def _otel_export(args: argparse.Namespace) -> int:
    traces = load_run_traces(args.run_dir)
    if args.jsonl is not None:
        write_jsonl(traces, args.jsonl, capture_content=args.capture_content)
        print(f"wrote {len(traces)} traces to {args.jsonl}")
        return EXIT_OK
    try:
        from .telemetry.otel_sdk import configure_tracer_provider, emit_traces
    except ImportError:
        print(
            'aire otel-export --endpoint needs the OpenTelemetry SDK: pip install "aire[otel]"',
            file=sys.stderr,
        )
        return EXIT_USAGE
    provider = configure_tracer_provider(
        service_name=args.service_name, otlp_endpoint=args.endpoint
    )
    emit_traces(traces, provider, capture_content=args.capture_content)
    flushed = provider.force_flush()
    provider.shutdown()
    print(f"sent {len(traces)} traces to {args.endpoint} (flushed: {flushed})")
    return EXIT_OK if flushed else EXIT_USAGE


def _otel_ingest(args: argparse.Namespace) -> int:
    traces = spans_to_traces(read_jsonl(args.spans), run_id=args.run_id)
    cases = load_eval_dataset(args.dataset)
    corpus = (
        {p.stem: p.read_text(encoding="utf-8") for p in sorted(args.corpus_dir.glob("*.md"))}
        if args.corpus_dir
        else None
    )
    run_id = args.run_id or (traces[0].run_id if traces else "otel_import")
    store = RunStore(args.out.parent, args.out.name, run_id)
    store.write_manifest(
        created_at=time.time(),
        system_config={"source": "opentelemetry", "spans": args.spans.name},
        dataset_path=args.dataset,
        n_cases=len(cases),
        deterministic=False,
        corpus=corpus,
    )
    for trace in traces:
        store.append_trace(trace)
    results = EvaluationEngine(corpus=corpus).evaluate_run(traces, cases)
    write_eval_results(store.run_dir, results)
    print(f"imported {len(traces)} traces into {store.run_dir} and evaluated them")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "gate":
        return run_gate(args.args)
    handlers = {
        "report": _report,
        "markdown": _markdown,
        "pr-comment": _pr_comment,
        "otel-export": _otel_export,
        "otel-ingest": _otel_ingest,
    }
    try:
        return handlers[args.command](args)
    except (OSError, ValueError, KeyError, GitHubError) as exc:  # json errors are ValueErrors
        print(f"aire {args.command}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_USAGE


def entry_point() -> None:
    raise SystemExit(main())
