"""The data behind the report's figures is generated from committed results.

scripts/figure_data.py writes reports/figures/data/; these tests fail the
build when that data no longer matches the committed results.
"""

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "figure_data", ROOT / "scripts" / "figure_data.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


figure_data = _load_script()


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def test_committed_figure_data_matches_the_committed_results() -> None:
    assert figure_data.sync(check=True) == [], "run python scripts/figure_data.py"


def test_deltas_are_the_gates_paired_intervals() -> None:
    comparison = json.loads(
        (ROOT / "results" / "v3_fluent_regression_vs_v1_baseline.json").read_text("utf-8")
    )
    rows = {row["metric"]: row for row in comparison["metrics"]}
    plotted = _rows(figure_data.delta_table(comparison))
    by_y = {int(r["y"]): r for r in plotted}
    count = len(figure_data.METRICS)
    for position, (metric, _) in enumerate(figure_data.METRICS):
        y = count - 1 - position
        row = rows[metric]
        if row.get("ci_low") is None:
            assert y not in by_y  # no interval, nothing plotted
            continue
        point = by_y[y]
        delta = float(point["delta"])
        assert abs(delta - row["paired_delta"]) < 1e-6
        assert abs(delta - float(point["minus"]) - row["ci_low"]) < 1e-6
        assert abs(delta + float(point["plus"]) - row["ci_high"]) < 1e-6
        assert point["verdict"] == row["verdict"]


def test_v3_citation_precision_is_left_out_because_it_has_no_interval() -> None:
    labels = {int(r["y"]): r["label"] for r in _rows(figure_data.label_table())}
    v3 = {int(r["y"]) for r in _rows(figure_data.render()["deltas_v3.csv"])}
    v2 = {int(r["y"]) for r in _rows(figure_data.render()["deltas_v2.csv"])}
    missing = {labels[y] for y in v2 - v3}
    assert missing == {r"citation\_precision"}


def test_runner_rows_are_the_benchmark_runs() -> None:
    benchmark = json.loads((ROOT / "results" / "runner_benchmark.json").read_text("utf-8"))
    runs = {run["label"]: run for run in benchmark["runs"]}
    rows = _rows(figure_data.runner_table(benchmark))
    assert len(rows) == len(runs) == len(figure_data.RUNS)
    for row in rows:
        key = next(k for k, label in figure_data.RUNS if label == row["label"])
        assert float(row["wall_s"]) == runs[key]["wall_time_s"]
        assert int(row["failed"]) == runs[key]["n_failed"]
