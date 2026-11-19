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
