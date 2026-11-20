"""Regression gate: compare two evaluated runs and exit non-zero on regression.

This is the piece that turns AIRE's run comparison into a CI check. A pipeline
evaluates the candidate system, then runs

    python -m aire.regression.gate BASELINE_RUN CANDIDATE_RUN

and the job fails when any gated metric regressed beyond its threshold.

Exit codes:
    0  no gated metric regressed
    1  at least one gated metric regressed (fail the build)
    2  usage error: a missing or malformed run directory, or two runs that
       did not evaluate the same dataset. Never 1, so a crash can not be
       mistaken for a blocked regression.

With --report, a gate that cannot compare because the evaluation set itself
changed (different dataset_sha256) still exits 2 but writes a
"not_comparable" report naming both dataset files and hashes, so CI can tell
a changed evaluation set from a crash. Every other exit 2 leaves no report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..tracing.storage import atomic_write_text
from .compare import (
    DatasetChangedError,
    RegressionThresholds,
    StatsConfig,
    compare_runs,
    write_regression_report,
)

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
        "candidate",
        type=Path,
        help="evaluated candidate run directory (contains eval_results.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help=(
            "default tolerance per metric (default: 0.02): an absolute difference "
            "for scores in [0, 1], a relative change for latency, tokens and cost"
        ),
    )
    parser.add_argument(
        "--override",
        type=_parse_override,
        action="append",
        default=[],
        metavar="METRIC=THRESHOLD",
        help="per-metric tolerance, e.g. latency_ms=0.5 (repeatable)",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="METRIC",
        help="report this metric but never fail on it (repeatable)",
    )
    parser.add_argument(
        "--method",
        choices=("paired", "point"),
        default="paired",
        help=(
            "paired (default): a change beyond tolerance must also be significant in a "
            "paired test over the cases; point: the tolerance alone (the Step 1 rule)"
        ),
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="significance level (default: 0.05)"
    )
    parser.add_argument(
        "--fail-on-inconclusive",
        action="store_true",
        help="also fail when a drop beyond tolerance is not statistically significant",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="also write the full comparison (with per-case attribution) as JSON",
    )
    return parser


def _num(value: Any, spec: str, width: int = 0) -> str:
    """Format a number, or '-' for a missing value, right-aligned to `width`."""
    return ("-" if value is None else format(value, spec)).rjust(width)


_TEST_LABELS = {"mcnemar_exact": "McNemar", "paired_bootstrap": "bootstrap", "point": "tolerance"}


def _format_row(row: dict[str, Any]) -> str:
    rel = "n/a" if row.get("rel_delta") is None else f"{row['rel_delta']:+.1%}"
    verdict = row["verdict"]
    if not row.get("gated", True):
        verdict += " (ignored)"
    if row.get("n_mismatch"):
        verdict += f" [n {row.get('n_baseline')} vs {row.get('n_candidate')}]"
    if row.get("direction_of_change"):
        verdict += f" ({row['direction_of_change']}, not significant)"
    test = _TEST_LABELS.get(str(row.get("test")), "")
    p_value = row.get("p_value")
    evidence = f"{test} p={p_value:.3g}" if p_value is not None else test
    return (
        f"  {row['metric']:<24} {_num(row['baseline'], '.4g', 10)} "
        f"{_num(row['candidate'], '.4g', 10)} {_num(row['delta'], '+.4g', 10)} {rel:>8}  "
        f"{evidence:<20} {verdict}"
    )


def dataset_changed_report(exc: DatasetChangedError) -> dict[str, Any]:
    """The gate report for two runs over different evaluation sets."""
    return {
        "status": "not_comparable",
        "reason": "dataset_changed",
        "dataset": {
            "baseline_path": exc.baseline_path,
            "baseline_sha256": exc.baseline_sha256,
            "candidate_path": exc.candidate_path,
            "candidate_sha256": exc.candidate_sha256,
        },
        "message": str(exc),
    }


def run_gate(argv: Sequence[str] | None = None) -> int:
    """Run the gate and return its exit code (see module docstring)."""
    args = build_parser().parse_args(argv)
    thresholds = RegressionThresholds(default=args.threshold, overrides=dict(args.override))
    ignored = set(args.ignore)
    if args.report is not None:
        # A report left from an earlier run must not describe this one.
        args.report.unlink(missing_ok=True)
    try:
        comparison = compare_runs(
            args.baseline,
            args.candidate,
            thresholds=thresholds,
            labels=(str(args.baseline), str(args.candidate)),
            ignore=ignored,
            stats=StatsConfig(method=args.method, alpha=args.alpha),
        )
    except DatasetChangedError as exc:
        print(f"aire-gate: cannot compare runs: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.report is not None:
            atomic_write_text(args.report, json.dumps(dataset_changed_report(exc), indent=2) + "\n")
        return EXIT_USAGE
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        # Any of these means the inputs are unusable; exit 2, never 1.
        print(f"aire-gate: cannot compare runs: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.report is not None:
        write_regression_report(comparison, args.report)

    rows = comparison["metrics"]
    blocking = [
        r
        for r in rows
        if r["gated"]
        and (
            r["verdict"] == "regressed"
            or (
                args.fail_on_inconclusive
                and r["verdict"] == "inconclusive"
                and r.get("direction_of_change") == "worse"
            )
        )
    ]

    print(f"aire-gate: {args.baseline} -> {args.candidate}")
    print(
        f"  {'metric':<24} {'baseline':>10} {'candidate':>10} {'delta':>10} {'rel':>8}  "
        f"{'evidence':<20} verdict"
    )
    for row in rows:
        print(_format_row(row))
    for warning in comparison.get("warnings", []):
        print(f"  warning: {warning}")

    for row in blocking:
        cases = row.get("attribution", [])
        if cases:
            worst = ", ".join(f"{c['case_id']} ({_num(c['delta'], '+.3g')})" for c in cases[:5])
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
