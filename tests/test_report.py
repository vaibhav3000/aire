"""Static HTML reports must escape every value taken from results files."""

from pathlib import Path

from aire.report import render_comparison_report, render_run_report

MARKUP = "<img src=x onerror=alert(1)>"


def _results() -> dict:
    row = {"case_id": MARKUP, "keyword_recall": 1.0, "failure": MARKUP, "answer": "fine"}
    return {"per_case": [row], "aggregates": {"n_cases": 1, "token_sources": [MARKUP]}}


def test_run_report_escapes_case_ids_and_failures(tmp_path: Path) -> None:
    out = render_run_report(tmp_path, _results(), tmp_path / "run.html", title="t")
    html = out.read_text(encoding="utf-8")
    assert MARKUP not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_comparison_report_escapes_metric_rows(tmp_path: Path) -> None:
    comparison = {
        "summary": {
            "verdict": "no regressions detected",
            "n_improved": 0,
            "n_regressed": 0,
            "n_neutral": 0,
        },
        "metrics": [
            {
                "metric": MARKUP,
                "direction": "higher",
                "baseline": MARKUP,
                "candidate": None,
                "delta": None,
                "verdict": "new",
            }
        ],
        "failure_count_deltas": {},
    }
    out = render_comparison_report(comparison, _results(), _results(), tmp_path / "cmp.html")
    assert MARKUP not in out.read_text(encoding="utf-8")


def test_case_table_shows_fixed_cutoff_retrieval_recall(tmp_path: Path) -> None:
    row = {"case_id": "c1", "retrieval_recall_at_3": 0.5, "failure": None, "answer": "x"}
    results = {"per_case": [row], "aggregates": {"n_cases": 1}}
    html = render_run_report(tmp_path, results, tmp_path / "run.html", title="t").read_text(
        encoding="utf-8"
    )
    assert "<th>retrieval_recall_at_3</th>" in html


ROOT = Path(__file__).resolve().parents[1]


def _committed(name: str) -> dict:
    import json

    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def _v3_report(tmp_path: Path) -> str:
    out = render_comparison_report(
        _committed("v3_fluent_regression_vs_v1_baseline.json"),
        _committed("v1_baseline/eval_results.json"),
        _committed("v3_fluent_regression/eval_results.json"),
        tmp_path / "v3.html",
        title="v3 vs v1",
    )
    return out.read_text(encoding="utf-8")


def test_diff_report_leads_with_the_decision_and_regressions_first(tmp_path: Path) -> None:
    html = _v3_report(tmp_path)
    assert "Blocked" in html and "regressions detected" in html
    body = html[html.index("<main") :]
    groundedness = body.index('id="metric-groundedness"')
    neutral = body.index('id="metric-retrieval_recall_at_1"')
    assert groundedness < neutral  # regressed rows come before neutral ones


def test_each_tested_metric_draws_its_interval_against_the_tolerance(tmp_path: Path) -> None:
    html = _v3_report(tmp_path)
    assert html.count('class="strip"') >= 5
    assert 'role="img"' in html
    assert "95% interval" in html and "tolerance" in html
    assert "McNemar" in html


def test_changed_cases_show_both_answers_most_affected_first(tmp_path: Path) -> None:
    html = _v3_report(tmp_path)
    comparison = _committed("v3_fluent_regression_vs_v1_baseline.json")
    moved: dict[str, int] = {}
    for row in comparison["metrics"]:
        if row["gated"]:
            for entry in row.get("attribution") or []:
                moved[entry["case_id"]] = moved.get(entry["case_id"], 0) + 1
    ordered = sorted(moved, key=lambda c: (-moved[c], c))
    assert ordered[0] in html
    shown = min(40, len(ordered))
    assert html.count('class="answer baseline"') == shown
    assert html.count('class="answer candidate"') == shown
    if len(ordered) > 40:
        assert f"{len(ordered) - 40} more cases moved" in html


def test_the_page_is_self_contained_responsive_and_themed(tmp_path: Path) -> None:
    html = _v3_report(tmp_path)
    assert '<meta name="viewport"' in html
    assert "prefers-color-scheme: dark" in html
    assert "prefers-reduced-motion" in html
    assert ":focus-visible" in html
    assert "<script src" not in html and "<link rel=\"stylesheet\"" not in html
    assert 'lang="en"' in html


def test_answers_and_case_ids_from_results_are_escaped(tmp_path: Path) -> None:
    comparison = _committed("v3_fluent_regression_vs_v1_baseline.json")
    base = _committed("v1_baseline/eval_results.json")
    cand = _committed("v3_fluent_regression/eval_results.json")
    cand["per_case"][0]["answer"] = MARKUP
    comparison["metrics"][0]["attribution"] = [
        {"case_id": MARKUP, "baseline": 1.0, "candidate": 0.0, "delta": -1.0}
    ]
    html = render_comparison_report(comparison, base, cand, tmp_path / "x.html").read_text(
        encoding="utf-8"
    )
    assert MARKUP not in html
