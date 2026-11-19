"""Paired statistics in compare_runs and the CI-aware gate (ADR 0004)."""

import json
from pathlib import Path
from typing import Any

from aire.regression import compare_runs
from aire.regression.compare import StatsConfig
from aire.regression.gate import EXIT_PASS, EXIT_REGRESSED, run_gate

SHA = "a" * 64


def _run(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write an evaluated run whose aggregates are the means of the per-case rows."""
    path.mkdir(parents=True, exist_ok=True)
    metrics = {k for row in rows for k, v in row.items() if isinstance(v, float)}
    aggregates: dict[str, Any] = {"n_cases": len(rows)}
    for m in sorted(metrics):
        values = [row[m] for row in rows if isinstance(row.get(m), float)]
        aggregates[f"mean_{m}"] = sum(values) / len(values)
        aggregates[f"n_{m}"] = len(values)
    if all("failure" in row for row in rows):
        ok = sum(1 for row in rows if row["failure"] is None)
        aggregates["success_rate"] = ok / len(rows)
    results = {"per_case": rows, "aggregates": aggregates}
    (path / "eval_results.json").write_text(json.dumps(results), encoding="utf-8")
    (path / "manifest.json").write_text(json.dumps({"dataset_sha256": SHA}), encoding="utf-8")
    return path


def _rows(n: int, values: list[float], metric: str = "groundedness") -> list[dict[str, Any]]:
    return [
        {"case_id": f"c{i:03d}", metric: values[i], "failure": None if values[i] else "ungrounded"}
        for i in range(n)
    ]


def _by_metric(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["metric"]: row for row in report["metrics"]}


def test_a_large_drop_on_a_binary_metric_is_a_significant_regression(tmp_path: Path) -> None:
    base = _run(tmp_path / "b", _rows(60, [1.0] * 60))
    cand = _run(tmp_path / "c", _rows(60, [1.0] * 40 + [0.0] * 20))
    row = _by_metric(compare_runs(base, cand))["groundedness"]
    assert row["verdict"] == "regressed"
    assert row["test"] == "mcnemar_exact"
    assert row["n_pairs"] == 60
    assert row["p_value"] < 1e-5
    assert row["ci_high"] < 0
    assert row["attribution"]


def test_a_drop_beyond_tolerance_without_evidence_is_inconclusive(tmp_path: Path) -> None:
    # 3 of 60 cases flip down: -0.05 is beyond the 0.02 tolerance, but McNemar
    # on 3 discordant pairs gives p = 0.25.
    base = _run(tmp_path / "b", _rows(60, [1.0] * 60))
    cand = _run(tmp_path / "c", _rows(60, [1.0] * 57 + [0.0] * 3))
    report = compare_runs(base, cand)
    row = _by_metric(report)["groundedness"]
    assert row["verdict"] == "inconclusive"
    assert row["direction_of_change"] == "worse"
    assert row["p_value"] == 0.25
    assert report["summary"]["n_inconclusive"] >= 1
    assert report["summary"]["n_regressed"] == 0


def test_continuous_metrics_use_the_paired_bootstrap(tmp_path: Path) -> None:
    values = [0.5 + 0.004 * i for i in range(50)]
    base = _run(tmp_path / "b", _rows(50, values, metric="keyword_recall"))
    cand = _run(tmp_path / "c", _rows(50, [v - 0.2 for v in values], metric="keyword_recall"))
    row = _by_metric(compare_runs(base, cand))["keyword_recall"]
    assert row["test"] == "paired_bootstrap"
    assert row["verdict"] == "regressed"
    assert row["ci_low"] <= row["delta"] <= row["ci_high"]


def test_fewer_than_ten_pairs_fall_back_to_the_tolerance_rule(tmp_path: Path) -> None:
    base = _run(tmp_path / "b", _rows(4, [1.0] * 4))
    cand = _run(tmp_path / "c", _rows(4, [1.0, 1.0, 1.0, 0.0]))
    row = _by_metric(compare_runs(base, cand))["groundedness"]
    assert row["test"] == "point"
    assert row["verdict"] == "regressed"


def test_point_method_reproduces_the_step_1_rule(tmp_path: Path) -> None:
    base = _run(tmp_path / "b", _rows(60, [1.0] * 60))
    cand = _run(tmp_path / "c", _rows(60, [1.0] * 57 + [0.0] * 3))
    report = compare_runs(base, cand, stats=StatsConfig(method="point"))
    assert _by_metric(report)["groundedness"]["verdict"] == "regressed"


def test_success_rate_is_paired_from_per_case_failures(tmp_path: Path) -> None:
    base = _run(tmp_path / "b", _rows(60, [1.0] * 60))
    cand = _run(tmp_path / "c", _rows(60, [1.0] * 40 + [0.0] * 20))
    row = _by_metric(compare_runs(base, cand))["success_rate"]
    assert row["test"] == "mcnemar_exact"
    assert row["verdict"] == "regressed"


def test_gated_rows_carry_holm_adjusted_p_values(tmp_path: Path) -> None:
    base = _run(tmp_path / "b", _rows(60, [1.0] * 60))
    cand = _run(tmp_path / "c", _rows(60, [1.0] * 40 + [0.0] * 20))
    for row in compare_runs(base, cand)["metrics"]:
        assert row["p_holm"] >= row["p_value"]


def test_the_gate_blocks_inconclusive_drops_only_when_asked(tmp_path: Path) -> None:
    base = str(_run(tmp_path / "b", _rows(60, [1.0] * 60)))
    cand = str(_run(tmp_path / "c", _rows(60, [1.0] * 57 + [0.0] * 3)))
    assert run_gate([base, cand]) == EXIT_PASS
    assert run_gate([base, cand, "--fail-on-inconclusive"]) == EXIT_REGRESSED
    assert run_gate([base, cand, "--method", "point"]) == EXIT_REGRESSED
    assert run_gate([base, cand, "--alpha", "0.3"]) == EXIT_REGRESSED  # p = 0.25 < 0.3
