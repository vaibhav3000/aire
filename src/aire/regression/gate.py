"""Regression gate: compare two evaluated runs and exit non-zero on regression.

This is the piece that turns AIRE's run comparison into a CI check. A pipeline
evaluates the candidate system, then runs

    python -m aire.regression.gate BASELINE_RUN CANDIDATE_RUN

and the job fails when any gated metric regressed beyond its threshold.

Exit codes:
    0  no gated metric regressed
    1  at least one gated metric regressed (fail the build)
    2  usage error, or a run directory without eval_results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from .compare import RegressionThresholds, compare_runs, write_regression_report

EXIT_PASS = 0
EXIT_REGRESSED = 1
EXIT_USAGE = 2


def _parse_override(text: str) -> tuple[str, float]:
    metric, sep, value = text.partition("=")
    if not sep or not metric.strip():
        raise argparse.ArgumentTypeError(f"expected METRIC=THRESHOLD, got {text!r}")
    try:
        return metric.strip(), float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"threshold for {metric.strip()!r} is not a number: {value!r}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aire-gate",
        description="Fail (exit 1) when a candidate run regresses against a baseline run.",
    )
    parser.add_argument(
        "baseline", type=Path, help="evaluated baseline run directory (contains eval_results.json)"
    )
    parser.add_argument(
        "candidate", type=Path, help="evaluated candidate run directory (contains eval_results.json)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.02,
        help="default relative threshold per metric (default: 0.02, i.e. 2%%)",
    )
    parser.add_argument(
        "--override", type=_parse_override, action="append", default=[],
        metavar="METRIC=THRESHOLD",
        help="per-metric relative threshold, e.g. latency_ms=0.5 (repeatable)",
    )
    parser.add_argument(
        "--ignore", action="append", default=[], metavar="METRIC",
        help="report this metric but never fail on it (repeatable)",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="also write the full comparison (with per-case attribution) as JSON",
    )
    return parser


def _format_row(row: dict[str, Any], ignored: set[str]) -> str:
    base, cand = row["baseline"], row["candidate"]
    # A relative change is meaningless against a zero baseline; show the absolute delta.
    rel = f"{row['rel_delta']:+.1%}" if abs(base) > 1e-9 else "n/a"
    verdict = row["verdict"]
    if verdict == "regressed" and row["metric"] in ignored:
        verdict += " (ignored)"
    return f"  {row['metric']:<20} {base:>10.4g} {cand:>10.4g} {row['delta']:>+10.4g} {rel:>8}  {verdict}"


def run_gate(argv: Sequence[str] | None = None) -> int:
    """Run the gate and return its exit code (see module docstring)."""
    args = build_parser().parse_args(argv)
    thresholds = RegressionThresholds(default=args.threshold, overrides=dict(args.override))
    try:
        comparison = compare_runs(
            args.baseline,
            args.candidate,
            thresholds=thresholds,
            labels=(str(args.baseline), str(args.candidate)),
        )
    except FileNotFoundError as exc:
        print(f"aire-gate: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.report is not None:
        write_regression_report(comparison, args.report)

    ignored = set(args.ignore)
    rows = comparison["metrics"]
    blocking = [r for r in rows if r["verdict"] == "regressed" and r["metric"] not in ignored]

    print(f"aire-gate: {args.baseline} -> {args.candidate}")
    print(f"  {'metric':<20} {'baseline':>10} {'candidate':>10} {'delta':>10} {'rel':>8}  verdict")
    for row in rows:
        print(_format_row(row, ignored))

    for row in blocking:
        cases = row.get("attribution", [])
        if cases:
            worst = ", ".join(f"{c['case_id']} ({c['delta']:+.3g})" for c in cases[:5])
            more = f" and {len(cases) - 5} more" if len(cases) > 5 else ""
            print(f"  {row['metric']}: {len(cases)} case(s) regressed, worst: {worst}{more}")

    if blocking:
        names = ", ".join(r["metric"] for r in blocking)
        print(f"GATE FAILED: {len(blocking)} metric(s) regressed: {names}")
        return EXIT_REGRESSED

    summary = comparison["summary"]
    print(
        f"GATE PASSED: no gated regressions "
        f"(improved={summary['n_improved']}, neutral={summary['n_neutral']}, "
        f"ignored={len(ignored)})"
    )
    return EXIT_PASS


def main() -> None:
    raise SystemExit(run_gate())


if __name__ == "__main__":
    main()
