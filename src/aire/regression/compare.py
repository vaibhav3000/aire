"""Regression engine: compares two evaluation runs and gates each metric.

The output answers the question a team actually asks after a change:
"what got better, what got worse, and by how much?", per metric, with
configurable thresholds and per-case attribution for every regression.

Comparison rules:
- Only runs over the same evaluation set are compared. When both runs have a
  manifest.json their dataset_sha256 must match, and the case ids must match
  in every case; otherwise compare_runs raises RunMismatchError.
- Bounded scores (in [0, 1]) are compared with an absolute tolerance;
  latency, tokens and cost with a tolerance relative to the baseline, which
  falls back to an absolute difference when the baseline is zero.
- A metric the baseline reports but the candidate does not is a regression:
  a broken candidate must not pass by producing fewer metrics.
- A mean over a different number of cases is flagged (n_mismatch).
- Token and cost means measured from different token sources (an API count
  against a character heuristic) get the verdict "not comparable".

Statistics (ADR 0004). Cases are paired by id. A change beyond the tolerance
only counts when a paired test supports it at alpha: McNemar's exact test
when the metric is 0 or 1 on every paired case, a seeded paired bootstrap of
the mean difference otherwise. A change beyond tolerance without that
support is "inconclusive". Rows with fewer than StatsConfig.min_pairs paired
values keep the tolerance rule alone (test "point"), because resampling a
handful of values gives no usable interval. Holm-adjusted p-values over the
gated rows are reported next to the raw ones; the verdict uses the raw p.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeGuard

from ..evaluator.metrics import metric_info
from ..evaluator.stats import holm_adjust, mcnemar_exact, paired_bootstrap
from ..tracing.storage import atomic_write_text

_TOKEN_METRICS = frozenset({"total_tokens", "cost_usd"})
_STANDALONE_METRICS = frozenset({"success_rate"})


class RunMismatchError(ValueError):
    """The two runs did not evaluate the same dataset, so they are not comparable."""


@dataclass
class RegressionThresholds:
    """Per-metric tolerances; metrics absent from `overrides` use `default`.

    For bounded scores the tolerance is an absolute difference (0.02 means two
    points on a 0-1 scale). For latency, tokens and cost it is a relative
    change against the baseline (0.02 means 2%), or an absolute difference
    when the baseline is zero. A change counts only beyond the tolerance, so
    tiny numerical noise does not produce noise reports.
    """

    default: float = 0.02
    overrides: dict[str, float] = field(default_factory=dict)

    def for_metric(self, metric: str) -> float:
        return self.overrides.get(metric, self.default)


@dataclass(frozen=True)
class StatsConfig:
    """How compare_runs decides significance; method "point" turns statistics off."""

    method: Literal["paired", "point"] = "paired"
    alpha: float = 0.05
    n_resamples: int = 2000
    seed: int = 0
    min_pairs: int = 10


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_eval_results(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "eval_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; evaluate both runs before comparing "
            f"(EvaluationEngine.evaluate_run + write_eval_results)"
        )
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _load_manifest(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        return None
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _is_number(value: object) -> TypeGuard[float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _metric_props(metric: str) -> tuple[str, bool]:
    """(direction, bounded) from the registry; unknown metrics compare relatively."""
    info = metric_info(metric) or {}
    return info.get("direction", "higher"), bool(info.get("bounded", False))


def _change(bv: float, cv: float, bounded: bool) -> float:
    """The quantity compared against the tolerance (absolute or relative)."""
    delta = cv - bv
    if bounded or bv == 0:
        return delta
    return delta / abs(bv)


def _verdict(change: float, direction: str, threshold: float) -> str:
    if direction == "lower":
        change = -change
    if change < -threshold:
        return "regressed"
    if change > threshold:
        return "improved"
    return "neutral"


def _check_same_dataset(
    base: dict[str, Any],
    cand: dict[str, Any],
    base_manifest: dict[str, Any] | None,
    cand_manifest: dict[str, Any] | None,
    warnings: list[str],
) -> None:
    if base_manifest is None or cand_manifest is None:
        warnings.append("dataset identity not verified: manifest.json missing in a run")
    else:
        base_sha = base_manifest.get("dataset_sha256")
        cand_sha = cand_manifest.get("dataset_sha256")
        if base_sha != cand_sha:
            raise RunMismatchError(
                f"runs evaluated different datasets (dataset_sha256 {base_sha} vs {cand_sha})"
            )
        base_corpus = base_manifest.get("corpus_sha256")
        cand_corpus = cand_manifest.get("corpus_sha256")
        if base_corpus and cand_corpus and base_corpus != cand_corpus:
            warnings.append(
                "the runs used different corpora (corpus_sha256 differs); "
                "groundedness is scored against each run's own documents"
            )
    base_ids = {row["case_id"] for row in base.get("per_case", [])}
    cand_ids = {row["case_id"] for row in cand.get("per_case", [])}
    if base_ids != cand_ids:
        only_base = sorted(base_ids - cand_ids)
        only_cand = sorted(cand_ids - base_ids)
        raise RunMismatchError(
            f"runs cover different case ids (only in baseline: {only_base}; "
            f"only in candidate: {only_cand})"
        )


def _case_value(row: dict[str, Any], metric: str) -> float | None:
    if metric == "success_rate":
        return None if "failure" not in row else float(row["failure"] is None)
    value = row.get(metric)
    return float(value) if _is_number(value) else None


def _pairs(
    metric: str,
    base_by_case: dict[str, dict[str, Any]],
    cand_by_case: dict[str, dict[str, Any]],
) -> tuple[list[float], list[float]]:
    base, cand = [], []
    for case_id in sorted(base_by_case.keys() & cand_by_case.keys()):
        bv = _case_value(base_by_case[case_id], metric)
        cv = _case_value(cand_by_case[case_id], metric)
        if bv is not None and cv is not None:
            base.append(bv)
            cand.append(cv)
    return base, cand


def _paired_test(base: list[float], cand: list[float], stats: StatsConfig) -> dict[str, Any]:
    """Paired delta, interval, p-value and the name of the test used."""
    bootstrap = paired_bootstrap(
        base,
        cand,
        confidence=1.0 - stats.alpha,
        n_resamples=stats.n_resamples,
        seed=stats.seed,
    )
    result: dict[str, Any] = {
        "test": "paired_bootstrap",
        "paired_delta": round(bootstrap.delta, 6),
        "ci_low": round(bootstrap.ci_low, 6),
        "ci_high": round(bootstrap.ci_high, 6),
        "p_value": round(bootstrap.p_value, 6),
    }
    if all(v in (0.0, 1.0) for v in base + cand):
        only_base = sum(1 for b, c in zip(base, cand, strict=True) if b > c)
        only_cand = sum(1 for b, c in zip(base, cand, strict=True) if c > b)
        result["test"] = "mcnemar_exact"
        result["p_value"] = round(mcnemar_exact(only_base, only_cand), 6)
        result["discordant"] = {"baseline_only": only_base, "candidate_only": only_cand}
    return result


def compare_runs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    thresholds: RegressionThresholds | None = None,
    labels: tuple[str, str] = ("baseline", "candidate"),
    ignore: Collection[str] = (),
    stats: StatsConfig | None = None,
) -> dict[str, Any]:
    """Compare two evaluated runs; produce a regression report dict.

    Every mean_<metric> aggregate and success_rate is compared with the
    direction and tolerance type from the metric registry. Metrics named in
    `ignore` are still reported (with gated=False) but never count towards
    the summary or the verdict. Regressions carry per-case attribution.
    Raises RunMismatchError when the runs are not over the same dataset.
    """
    thresholds = thresholds or RegressionThresholds()
    stats = stats or StatsConfig()
    base = _load_eval_results(baseline_dir)
    cand = _load_eval_results(candidate_dir)
    warnings: list[str] = []
    _check_same_dataset(
        base, cand, _load_manifest(baseline_dir), _load_manifest(candidate_dir), warnings
    )

    base_agg: dict[str, Any] = base.get("aggregates", {})
    cand_agg: dict[str, Any] = cand.get("aggregates", {})
    base_by_case = {row["case_id"]: row for row in base.get("per_case", [])}
    cand_by_case = {row["case_id"]: row for row in cand.get("per_case", [])}
    token_sources_match = (
        "token_sources" in base_agg
        and "token_sources" in cand_agg
        and base_agg["token_sources"] == cand_agg["token_sources"]
    )

    metric_rows: list[dict[str, Any]] = []
    for key in sorted(set(base_agg) | set(cand_agg)):
        if not (key.startswith("mean_") or key in _STANDALONE_METRICS):
            continue
        metric = key.removeprefix("mean_")
        direction, bounded = _metric_props(metric)
        threshold = thresholds.for_metric(metric)
        bv, cv = base_agg.get(key), cand_agg.get(key)
        row: dict[str, Any] = {
            "metric": metric,
            "direction": direction,
            "tolerance": "absolute" if bounded else "relative",
            "baseline": round(bv, 4) if _is_number(bv) else None,
            "candidate": round(cv, 4) if _is_number(cv) else None,
            "delta": None,
            "rel_delta": None,
            "threshold": threshold,
            "gated": metric not in ignore,
        }
        if _is_number(bv) and _is_number(cv):
            row["delta"] = round(cv - bv, 4)
            row["rel_delta"] = round((cv - bv) / abs(bv), 4) if bv != 0 else None
            if metric in _TOKEN_METRICS and not token_sources_match:
                row["verdict"] = "not comparable"
                row["note"] = "token counts come from different sources"
            else:
                row["verdict"] = _verdict(_change(bv, cv, bounded), direction, threshold)
                _apply_statistics(row, metric, base_by_case, cand_by_case, direction, stats)
        elif _is_number(bv):
            row["verdict"] = "regressed"
            row["note"] = "missing in candidate"
        else:
            row["verdict"] = "new"
            row["note"] = "missing in baseline"

        n_key = "n_cases" if metric in _STANDALONE_METRICS else f"n_{metric}"
        nb, nc = base_agg.get(n_key), cand_agg.get(n_key)
        if nb is not None or nc is not None:
            # A run that reports no mean for the metric averaged zero cases.
            nb = 0 if nb is None and not _is_number(bv) else nb
            nc = 0 if nc is None and not _is_number(cv) else nc
            row["n_baseline"], row["n_candidate"] = nb, nc
            if nb != nc:
                row["n_mismatch"] = True
                warnings.append(
                    f"{metric}: baseline averages {nb} case(s), candidate {nc}; "
                    "the means cover different cases"
                )
        if row["verdict"] == "regressed" or row.get("direction_of_change") == "worse":
            row["attribution"] = _attribute_regressions(
                metric, base_by_case, cand_by_case, direction, bounded, threshold
            )
        metric_rows.append(row)

    tested = [row for row in metric_rows if row["gated"] and row.get("p_value") is not None]
    for row, p_holm in zip(tested, holm_adjust([r["p_value"] for r in tested]), strict=True):
        row["p_holm"] = round(p_holm, 6)
    for row in metric_rows:
        row.setdefault("p_value", None)
        row.setdefault("p_holm", None)

    gated = [row["verdict"] for row in metric_rows if row["gated"]]
    summary = {
        "labels": {"baseline": labels[0], "candidate": labels[1]},
        "n_improved": gated.count("improved"),
        "n_regressed": gated.count("regressed"),
        "n_neutral": gated.count("neutral"),
        "n_inconclusive": gated.count("inconclusive"),
        "n_not_comparable": gated.count("not comparable"),
        "n_new": gated.count("new"),
        "n_ignored": sum(1 for row in metric_rows if not row["gated"]),
        "verdict": (
            "regressions detected" if gated.count("regressed") else "no regressions detected"
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
        "statistics": {
            "method": stats.method,
            "alpha": stats.alpha,
            "n_resamples": stats.n_resamples,
            "seed": stats.seed,
            "min_pairs": stats.min_pairs,
        },
        "metrics": metric_rows,
        "failure_count_deltas": failure_delta,
        "warnings": warnings,
    }


def _apply_statistics(
    row: dict[str, Any],
    metric: str,
    base_by_case: dict[str, dict[str, Any]],
    cand_by_case: dict[str, dict[str, Any]],
    direction: str,
    stats: StatsConfig,
) -> None:
    """Add the paired test to a row; a change the test does not support becomes inconclusive."""
    base, cand = _pairs(metric, base_by_case, cand_by_case)
    row["n_pairs"] = len(base)
    if stats.method == "point" or len(base) < stats.min_pairs:
        row["test"] = "point"
        return
    row.update(_paired_test(base, cand, stats))
    if row["verdict"] not in ("regressed", "improved"):
        return
    paired_better = (row["paired_delta"] > 0) == (direction == "higher")
    agrees = row["paired_delta"] != 0 and paired_better == (row["verdict"] == "improved")
    if row["p_value"] < stats.alpha and agrees:
        return
    row["direction_of_change"] = "better" if row["verdict"] == "improved" else "worse"
    row["verdict"] = "inconclusive"


def _attribute_regressions(
    metric: str,
    base_by_case: dict[str, dict[str, Any]],
    cand_by_case: dict[str, dict[str, Any]],
    direction: str,
    bounded: bool,
    threshold: float,
) -> list[dict[str, Any]]:
    """List the cases that individually regressed, using the aggregate's rule."""
    attribution: list[dict[str, Any]] = []
    for case_id, cand_row in cand_by_case.items():
        bv = base_by_case.get(case_id, {}).get(metric)
        cv = cand_row.get(metric)
        if not _is_number(bv):
            continue
        if not _is_number(cv):
            attribution.append(
                {"case_id": case_id, "baseline": bv, "candidate": None, "delta": None}
            )
            continue
        if _verdict(_change(bv, cv, bounded), direction, threshold) == "regressed":
            attribution.append(
                {"case_id": case_id, "baseline": bv, "candidate": cv, "delta": round(cv - bv, 4)}
            )
    # Most negative change first for higher-better metrics, largest rise first
    # for lower-better ones; cases that lost the metric entirely go last.
    sign = 1 if direction == "higher" else -1
    attribution.sort(
        key=lambda a: (a["delta"] is None, sign * a["delta"] if a["delta"] is not None else 0)
    )
    return attribution


def write_regression_report(report: dict[str, Any], out_path: str | Path) -> Path:
    return atomic_write_text(out_path, json.dumps(report, indent=2))
