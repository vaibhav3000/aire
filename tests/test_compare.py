"""Regression comparison: tolerances, missing metrics, run identity, attribution."""

import json
from pathlib import Path
from typing import Any

import pytest

from aire.regression import RegressionThresholds, compare_runs

SHA = "a" * 64


def _write_run(
    path: Path,
    aggregates: dict[str, Any],
    per_case: list[dict[str, Any]] | None = None,
    manifest: dict[str, Any] | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    rows = per_case if per_case is not None else [{"case_id": "c1"}, {"case_id": "c2"}]
    results = {"per_case": rows, "aggregates": {"n_cases": len(rows), **aggregates}}
    (path / "eval_results.json").write_text(json.dumps(results), encoding="utf-8")
    meta = {"dataset_sha256": SHA} if manifest is None else manifest
    (path / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def _rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["metric"]: row for row in report["metrics"]}


def test_metric_missing_from_the_candidate_is_a_regression(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", {"mean_groundedness": 1.0, "n_groundedness": 2})
    cand = _write_run(tmp_path / "cand", {})
    report = compare_runs(base, cand)
    row = _rows(report)["groundedness"]
    assert row["verdict"] == "regressed"
    assert row["candidate"] is None
    assert report["summary"]["n_regressed"] == 1


def test_bounded_scores_use_an_absolute_tolerance(tmp_path: Path) -> None:
    # -0.015 is under two points, although it is -15% relative to 0.10.
    base = _write_run(tmp_path / "base", {"mean_keyword_recall": 0.10})
    cand = _write_run(tmp_path / "cand", {"mean_keyword_recall": 0.085})
    assert _rows(compare_runs(base, cand))["keyword_recall"]["verdict"] == "neutral"


def test_unbounded_metrics_use_a_relative_tolerance(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", {"mean_latency_ms": 100.0})
    small = _write_run(tmp_path / "small", {"mean_latency_ms": 101.5})
    large = _write_run(tmp_path / "large", {"mean_latency_ms": 103.0})
    assert _rows(compare_runs(base, small))["latency_ms"]["verdict"] == "neutral"
    assert _rows(compare_runs(base, large))["latency_ms"]["verdict"] == "regressed"


def test_zero_baseline_falls_back_to_an_absolute_delta(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", {"mean_cost_usd": 0.0, "token_sources": ["api"]})
    cand = _write_run(tmp_path / "cand", {"mean_cost_usd": 0.000001, "token_sources": ["api"]})
    row = _rows(compare_runs(base, cand))["cost_usd"]
    assert row["verdict"] == "neutral"
    assert row["rel_delta"] is None


def test_runs_on_different_datasets_are_refused(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", {"mean_keyword_recall": 0.5})
    cand = _write_run(
        tmp_path / "cand", {"mean_keyword_recall": 0.5}, manifest={"dataset_sha256": "b" * 64}
    )
    with pytest.raises(ValueError, match="dataset"):
        compare_runs(base, cand)


def test_runs_over_different_case_sets_are_refused(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", {"mean_keyword_recall": 0.5})
    cand = _write_run(
        tmp_path / "cand",
        {"mean_keyword_recall": 0.5},
        per_case=[{"case_id": "c1"}, {"case_id": "c3"}],
    )
    with pytest.raises(ValueError, match="case"):
        compare_runs(base, cand)


def test_a_mean_over_a_different_number_of_cases_is_flagged(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", {"mean_groundedness": 0.9, "n_groundedness": 22})
    cand = _write_run(tmp_path / "cand", {"mean_groundedness": 1.0, "n_groundedness": 15})
    report = compare_runs(base, cand)
    assert _rows(report)["groundedness"]["n_mismatch"] is True
    assert any("groundedness" in warning for warning in report["warnings"])


def test_tokens_from_different_sources_get_no_verdict(tmp_path: Path) -> None:
    base = _write_run(
        tmp_path / "base", {"mean_total_tokens": 83.0, "token_sources": ["heuristic"]}
    )
    cand = _write_run(tmp_path / "cand", {"mean_total_tokens": 605.0, "token_sources": ["api"]})
    report = compare_runs(base, cand)
    assert _rows(report)["total_tokens"]["verdict"] == "not comparable"
    assert report["summary"]["n_regressed"] == 0


def test_ignored_metrics_are_reported_but_not_counted(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", {"mean_latency_ms": 100.0})
    cand = _write_run(tmp_path / "cand", {"mean_latency_ms": 200.0})
    report = compare_runs(base, cand, ignore={"latency_ms"})
    row = _rows(report)["latency_ms"]
    assert row["verdict"] == "regressed"
    assert row["gated"] is False
    assert report["summary"]["n_regressed"] == 0
    assert report["summary"]["verdict"] == "no regressions detected"


def test_success_rate_and_error_rate_are_gated(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", {"success_rate": 1.0, "mean_errors": 0.0})
    cand = _write_run(tmp_path / "cand", {"success_rate": 0.5, "mean_errors": 0.5})
    rows = _rows(compare_runs(base, cand))
    assert rows["success_rate"]["verdict"] == "regressed"
    assert rows["errors"]["verdict"] == "regressed"


def test_case_counts_are_not_compared_as_metrics(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", {"mean_groundedness": 1.0, "n_groundedness": 22})
    cand = _write_run(tmp_path / "cand", {"mean_groundedness": 1.0, "n_groundedness": 22})
    assert set(_rows(compare_runs(base, cand))) == {"groundedness"}


def test_per_case_attribution_uses_the_same_rule_as_the_aggregate(tmp_path: Path) -> None:
    per_base = [{"case_id": "c1", "latency_ms": 100.0}, {"case_id": "c2", "latency_ms": 1000.0}]
    per_cand = [{"case_id": "c1", "latency_ms": 150.0}, {"case_id": "c2", "latency_ms": 1010.0}]
    base = _write_run(tmp_path / "base", {"mean_latency_ms": 550.0}, per_case=per_base)
    cand = _write_run(tmp_path / "cand", {"mean_latency_ms": 580.0}, per_case=per_cand)
    report = compare_runs(base, cand, thresholds=RegressionThresholds(default=0.02))
    attribution = _rows(report)["latency_ms"]["attribution"]
    assert [entry["case_id"] for entry in attribution] == ["c1"]
