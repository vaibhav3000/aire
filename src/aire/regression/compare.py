"""Regression engine: compares two evaluation runs and gates each metric.

The output answers the question a team actually asks after a change:
"what got better, what got worse, and by how much?" — per metric, with
configurable thresholds and per-case attribution for every regression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..evaluator.metrics import metric_spec


@dataclass
class RegressionThresholds:
    """Per-metric relative thresholds; metrics absent here use `default`.

    A delta counts as a regression only if it exceeds the threshold, which
    keeps tiny numerical noise from producing noise reports.
    """

    default: float = 0.02
    overrides: dict[str, float] = field(default_factory=dict)

    def for_metric(self, metric: str) -> float:
        return self.overrides.get(metric, self.default)


def _load_eval_results(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "eval_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — evaluate both runs before comparing "
            f"(EvaluationEngine.evaluate_run + write_eval_results)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def compare_runs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    thresholds: RegressionThresholds | None = None,
    labels: tuple[str, str] = ("baseline", "candidate"),
) -> dict[str, Any]:
    """Compare two evaluated runs; produce a regression report dict.

    Every aggregate metric that appears in both runs is compared with the
    direction from the metric registry (higher-better vs lower-better).
    Regressions carry per-case attribution: which cases moved, by how much.
    """
    thresholds = thresholds or RegressionThresholds()
    base = _load_eval_results(baseline_dir)
    cand = _load_eval_results(candidate_dir)
    spec = metric_spec()

    base_agg: dict[str, Any] = base.get("aggregates", {})
    cand_agg: dict[str, Any] = cand.get("aggregates", {})
    base_by_case = {row["case_id"]: row for row in base.get("per_case", [])}
    cand_by_case = {row["case_id"]: row for row in cand.get("per_case", [])}

    metric_rows: list[dict[str, Any]] = []
    for key in sorted(set(base_agg) | set(cand_agg)):
        if key.startswith("failure_counts") or key in ("n_cases", "n_success"):
            continue
        bv, cv = base_agg.get(key), cand_agg.get(key)
        if not isinstance(bv, (int, float)) or not isinstance(cv, (int, float)):
            continue
        metric = key.removeprefix("mean_")
        direction = spec.get(metric, {}).get("direction", "higher")
        # Relative change vs the baseline, so thresholds are scale-free
        # (a +0.6 token cost change on ~83 tokens is noise; a -0.57 drop in
        # groundedness from 0.57 is a catastrophe).
        denom = max(abs(bv), 1e-9)
        rel_delta = round((cv - bv) / denom, 4)
        threshold = thresholds.for_metric(metric)
        if direction == "higher":
            verdict = (
                "regressed" if rel_delta < -threshold
                else ("improved" if rel_delta > threshold else "neutral")
            )
        else:
            verdict = (
                "regressed" if rel_delta > threshold
                else ("improved" if rel_delta < -threshold else "neutral")
            )
        row = {
            "metric": metric,
            "direction": direction,
            "baseline": round(bv, 4),
            "candidate": round(cv, 4),
            "delta": round(cv - bv, 4),
            "rel_delta": rel_delta,
            "threshold": threshold,
            "verdict": verdict,
        }
        if verdict == "regressed":
            row["attribution"] = _attribute_regressions(
                metric, base_by_case, cand_by_case, direction, threshold
            )
        metric_rows.append(row)

    verdicts = [row["verdict"] for row in metric_rows]
    summary = {
        "labels": {"baseline": labels[0], "candidate": labels[1]},
        "n_improved": verdicts.count("improved"),
        "n_regressed": verdicts.count("regressed"),
        "n_neutral": verdicts.count("neutral"),
        "verdict": (
            "regressions detected" if verdicts.count("regressed") else "no regressions detected"
        ),
    }
    failure_base: dict[str, int] = base_agg.get("failure_counts", {})
    failure_cand: dict[str, int] = cand_agg.get("failure_counts", {})
    failure_delta = {
        cat: failure_cand.get(cat, 0) - failure_base.get(cat, 0)
        for cat in sorted(set(failure_base) | set(failure_cand))
        if failure_cand.get(cat, 0) != failure_base.get(cat, 0)
    }
    return {
        "summary": summary,
        "metrics": metric_rows,
        "failure_count_deltas": failure_delta,
    }


def _attribute_regressions(
    metric: str,
    base_by_case: dict[str, dict[str, Any]],
    cand_by_case: dict[str, dict[str, Any]],
    direction: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """List the cases that individually regressed on this metric."""
    attribution: list[dict[str, Any]] = []
    for case_id, cand_row in cand_by_case.items():
        base_row = base_by_case.get(case_id, {})
        bv, cv = base_row.get(metric), cand_row.get(metric)
        if not isinstance(bv, (int, float)) or not isinstance(cv, (int, float)):
            continue
        delta = cv - bv
        worse = delta < -threshold if direction == "higher" else delta > threshold
        if worse:
            attribution.append(
                {"case_id": case_id, "baseline": bv, "candidate": cv, "delta": round(delta, 4)}
            )
    attribution.sort(key=lambda a: a["delta"])
    return attribution


def write_regression_report(report: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
