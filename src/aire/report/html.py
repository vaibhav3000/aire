"""Static HTML reports: one self-contained file per run or per comparison.

Deliberately boring technology: no server, no script, no external font or
stylesheet, so a report can be attached to a pull request or a release and
opened anywhere. The comparison report answers, in order: can the candidate
ship, which metrics moved and how strong is the evidence, which failure
categories changed, and which cases moved (with both answers side by side).

Each metric row draws its paired change as a dot with its 95% interval
against the shaded tolerance band around zero, the evidence the gate uses.
Every value taken from a results file is escaped. Light and dark themes
follow the system setting; tables scroll inside their own container on
narrow screens.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .. import __version__
from ..tracing.storage import atomic_write_text

_VERDICT_ORDER = {
    "regressed": 0,
    "inconclusive": 1,
    "improved": 2,
    "not comparable": 3,
    "new": 4,
    "neutral": 5,
}
_TEST_LABELS = {
    "mcnemar_exact": "McNemar",
    "paired_bootstrap": "paired bootstrap",
    "point": "tolerance only",
}
_CASE_COLUMNS = [
    "case_id",
    "keyword_recall",
    "gold_citation_recall",
    "groundedness",
    "retrieval_recall_at_1",
    "retrieval_recall_at_3",
    "tool_selection_accuracy",
    "step_efficiency",
    "latency_ms",
    "total_tokens",
    "failure",
]

_CSS = """
:root {
  color-scheme: light dark;
  --paper: #f6f7f9; --surface: #ffffff; --ink: #1c2430; --muted: #5b6675;
  --rule: #d9dee5; --band: #e6ebf1; --zero: #8b95a3;
  --bad: #b4232c; --good: #0f7b5f; --unsure: #8a5a00; --plain: #5b6675;
  --bad-bg: #fbeceb; --good-bg: #e7f4ef; --unsure-bg: #fbf2df;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #12161c; --surface: #1a2028; --ink: #e6eaf0; --muted: #9aa5b4;
    --rule: #2b333e; --band: #26303c; --zero: #6c7787;
    --bad: #f0716f; --good: #3cc9a0; --unsure: #e0b341; --plain: #9aa5b4;
    --bad-bg: #3a1d20; --good-bg: #14302a; --unsure-bg: #352a12;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
}
.skip { position: absolute; left: -999px; }
.skip:focus { left: 16px; top: 16px; background: var(--surface); padding: 6px 10px; }
a { color: inherit; }
:focus-visible { outline: 3px solid var(--ink); outline-offset: 2px; }
main, header.top, footer { max-width: 1100px; margin: 0 auto; padding: 0 16px; }
header.top { padding-top: 32px; }
h1 { font-size: 1.5rem; line-height: 1.25; margin: 0 0 4px; text-wrap: balance; }
h2 { font-size: 1.15rem; margin: 40px 0 8px; scroll-margin-top: 16px; }
h3 { font-size: 1rem; margin: 20px 0 6px; }
p { max-width: 72ch; }
.meta { color: var(--muted); font-size: 0.9rem; margin: 0; }
.decision {
  margin: 20px 0 0; padding: 18px 20px; border-radius: 10px; background: var(--surface);
  border: 1px solid var(--rule); border-left: 6px solid var(--plain);
}
.decision.blocked { border-left-color: var(--bad); }
.decision.passed { border-left-color: var(--good); }
.decision strong { display: block; font-size: 1.6rem; line-height: 1.2; }
.decision.blocked strong { color: var(--bad); }
.decision.passed strong { color: var(--good); }
.decision p { margin: 6px 0 0; }
.scroll { overflow-x: auto; border: 1px solid var(--rule); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 10px; text-align: left; vertical-align: middle; border-bottom: 1px solid var(--rule); }
th { font-weight: 600; color: var(--muted); white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
td.num, th.num { text-align: right; white-space: nowrap; }
td.metric { font-weight: 600; min-width: 12rem; }
td.metric small { display: block; font-weight: 400; color: var(--muted); }
tr.ignored td { color: var(--muted); }
.verdict { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; white-space: nowrap; }
.v-regressed { color: var(--bad); background: var(--bad-bg); }
.v-improved { color: var(--good); background: var(--good-bg); }
.v-inconclusive { color: var(--unsure); background: var(--unsure-bg); }
.v-neutral, .v-new, .v-not-comparable { color: var(--plain); background: var(--band); }
.evidence { color: var(--muted); font-size: 0.85rem; min-width: 10rem; }
svg.strip { display: block; width: 168px; height: 30px; }
svg.strip .band { fill: var(--band); }
svg.strip .zero { stroke: var(--zero); stroke-width: 1; }
svg.strip .ci { stroke: var(--ink); stroke-width: 2; }
svg.strip .dot { stroke: var(--surface); stroke-width: 1.5; }
svg.strip .dot.regressed { fill: var(--bad); }
svg.strip .dot.improved { fill: var(--good); }
svg.strip .dot.inconclusive { fill: var(--unsure); }
svg.strip .dot.neutral, svg.strip .dot.other { fill: var(--plain); }
.cases details { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; margin: 8px 0; }
.cases summary { cursor: pointer; padding: 10px 14px; overflow-wrap: anywhere; }
.cases summary:hover { background: var(--band); }
.cases .moved { color: var(--muted); font-size: 0.9rem; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 0 14px 14px; }
.answer { margin: 0; padding: 10px 12px; border-radius: 8px; background: var(--paper); white-space: pre-wrap; overflow-wrap: anywhere; font-size: 0.92rem; }
.answer.candidate { border-left: 3px solid var(--unsure); }
.answer.baseline { border-left: 3px solid var(--rule); }
.label { display: block; font-size: 0.8rem; color: var(--muted); margin-bottom: 4px; }
ul.warnings { padding-left: 20px; }
footer { padding: 32px 16px 48px; color: var(--muted); font-size: 0.85rem; }
@media (max-width: 640px) {
  .pair { grid-template-columns: 1fr; }
  .decision strong { font-size: 1.3rem; }
}
@media (prefers-reduced-motion: reduce) {
  * { scroll-behavior: auto; transition: none; animation: none; }
}
"""


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _fmt(value: Any) -> str:
    """A value for an HTML cell, always escaped (case ids and labels come from files)."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    return _esc(value)


def _breakable(name: str) -> str:
    """An escaped identifier that may wrap after its underscores, never inside a word."""
    return "_<wbr>".join(_esc(part) for part in name.split("_"))


def _value_or_none(value: Any) -> str:
    return "none" if value is None else _num(value)


def _num(value: Any, spec: str = ".4g") -> str:
    return (
        "-"
        if not isinstance(value, int | float) or isinstance(value, bool)
        else format(value, spec)
    )


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="theme-color" content="#f6f7f9" media="(prefers-color-scheme: light)">\n'
        '<meta name="theme-color" content="#12161c" media="(prefers-color-scheme: dark)">\n'
        f"<title>{_esc(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        '<a class="skip" href="#metrics">Skip to the metrics</a>\n'
        f"{body}\n"
        f"<footer>Static report generated by aire {_esc(__version__)}. "
        "No server, script or network access is needed to read it.</footer>\n"
        "</body>\n</html>\n"
    )


# --- comparison report ------------------------------------------------------


def _band(row: dict[str, Any]) -> float:
    threshold = float(row.get("threshold") or 0.0)
    if row.get("tolerance") == "relative" and isinstance(row.get("baseline"), int | float):
        return threshold * abs(float(row["baseline"])) or threshold
    return threshold


def _strip(row: dict[str, Any]) -> str:
    """The forest-plot strip: paired change, 95% interval, tolerance band, zero line."""
    delta, low, high = row.get("paired_delta"), row.get("ci_low"), row.get("ci_high")
    if not all(isinstance(v, int | float) for v in (delta, low, high)):
        return '<span class="evidence">no paired interval</span>'
    assert isinstance(delta, int | float) and isinstance(low, int | float)
    assert isinstance(high, int | float)
    band = _band(row)
    reach = max(abs(low), abs(high), abs(delta), band) * 1.2 or 1e-9

    def x(v: float) -> float:
        return round(100 + v / reach * 90, 2)

    verdict = (
        row["verdict"]
        if row["verdict"] in ("regressed", "improved", "inconclusive", "neutral")
        else "other"
    )
    label = (
        f"change {delta:+.4g}, 95% interval {low:.4g} to {high:.4g}, "
        f"tolerance plus or minus {band:.4g}"
    )
    return (
        f'<svg class="strip" viewBox="0 0 200 30" role="img" aria-label="{_esc(label)}">'
        f"<title>{_esc(label)}</title>"
        f'<rect class="band" x="{x(-band)}" y="4" width="{max(0.5, x(band) - x(-band))}" height="22" rx="2"/>'
        f'<line class="zero" x1="100" y1="2" x2="100" y2="28"/>'
        f'<line class="ci" x1="{x(low)}" y1="15" x2="{x(high)}" y2="15"/>'
        f'<line class="ci" x1="{x(low)}" y1="10" x2="{x(low)}" y2="20"/>'
        f'<line class="ci" x1="{x(high)}" y1="10" x2="{x(high)}" y2="20"/>'
        f'<circle class="dot {verdict}" cx="{x(delta)}" cy="15" r="5"/></svg>'
    )


def _evidence(row: dict[str, Any]) -> str:
    test = row.get("test")
    if test is None:
        return _esc(row.get("note") or "")
    label = _TEST_LABELS.get(str(test), str(test))
    if test == "point":
        return f"{label} ({_fmt(row.get('n_pairs'))} pairs)"
    p = row.get("p_value")
    p_text = "p < 0.001" if isinstance(p, int | float) and p < 0.001 else f"p = {_num(p, '.3g')}"
    return f"{_esc(label)}, {p_text}, {_fmt(row.get('n_pairs'))} pairs"


def _verdict_badge(row: dict[str, Any]) -> str:
    verdict = str(row["verdict"])
    text = verdict
    if row.get("direction_of_change"):
        text = f"{verdict} ({row['direction_of_change']})"
    if not row.get("gated", True):
        text += ", not gated"
    css = verdict.replace(" ", "-")
    return f'<span class="verdict v-{_esc(css)}">{_esc(text)}</span>'


def _metric_rows(rows: list[dict[str, Any]]) -> str:
    def order(row: dict[str, Any]) -> tuple[int, int, int, str]:
        worse = 0 if row.get("direction_of_change") == "worse" else 1
        return (
            0 if row.get("gated", True) else 1,
            _VERDICT_ORDER.get(str(row["verdict"]), 9),
            worse,
            str(row["metric"]),
        )

    out = []
    for row in sorted(rows, key=order):
        metric = str(row["metric"])
        direction = f"{row.get('direction', 'higher')} is better"
        css = "" if row.get("gated", True) else ' class="ignored"'
        out.append(
            f'<tr id="metric-{_esc(metric)}"{css}>'
            f'<td class="metric">{_breakable(metric)}<small>{_esc(direction)}</small></td>'
            f'<td class="num">{_fmt(row.get("baseline"))}</td>'
            f'<td class="num">{_fmt(row.get("candidate"))}</td>'
            f'<td class="num">{_esc(_num(row.get("delta"), "+.4g"))}</td>'
            f"<td>{_strip(row)}</td>"
            f'<td class="evidence">{_evidence(row)}</td>'
            f"<td>{_verdict_badge(row)}</td></tr>"
        )
    return "".join(out)


def _failure_table(base: dict[str, Any], cand: dict[str, Any]) -> str:
    b = base.get("aggregates", {}).get("failure_counts", {})
    c = cand.get("aggregates", {}).get("failure_counts", {})
    categories = sorted(set(b) | set(c), key=lambda k: (-abs(c.get(k, 0) - b.get(k, 0)), k))
    if not categories:
        return "<p>No failures in either run.</p>"
    rows = "".join(
        f"<tr><td>{_esc(k)}</td><td class='num'>{_fmt(b.get(k, 0))}</td>"
        f"<td class='num'>{_fmt(c.get(k, 0))}</td>"
        f"<td class='num'>{c.get(k, 0) - b.get(k, 0):+d}</td></tr>"
        for k in categories
    )
    return (
        '<div class="scroll" role="region" aria-label="Failure categories" tabindex="0">'
        "<table><thead><tr><th>Category</th><th class='num'>Baseline</th>"
        "<th class='num'>Candidate</th><th class='num'>Change</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _changed_cases(
    rows: list[dict[str, Any]], base: dict[str, Any], cand: dict[str, Any], limit: int = 40
) -> str:
    moved: dict[str, list[str]] = {}
    for row in rows:
        if not row.get("gated", True):
            continue
        for entry in row.get("attribution") or []:
            moved.setdefault(str(entry["case_id"]), []).append(
                f"{row['metric']} {_value_or_none(entry.get('baseline'))} to "
                f"{_value_or_none(entry.get('candidate'))}"
            )
    if not moved:
        return "<p>No case moved beyond the tolerance on a gated metric.</p>"
    base_rows = {str(r["case_id"]): r for r in base.get("per_case", [])}
    cand_rows = {str(r["case_id"]): r for r in cand.get("per_case", [])}
    ordered = sorted(moved, key=lambda case_id: (-len(moved[case_id]), case_id))
    parts = []
    for case_id in ordered[:limit]:
        b, c = base_rows.get(case_id, {}), cand_rows.get(case_id, {})
        parts.append(
            "<details><summary>"
            f"<strong>{_esc(case_id)}</strong> "
            f'<span class="moved">{_esc("; ".join(moved[case_id]))}</span></summary>'
            '<div class="pair">'
            f'<div><span class="label">Baseline answer, failure: {_fmt(b.get("failure") or "none")}</span>'
            f'<p class="answer baseline">{_esc(b.get("answer") or "(no answer)")}</p></div>'
            f'<div><span class="label">Candidate answer, failure: {_fmt(c.get("failure") or "none")}</span>'
            f'<p class="answer candidate">{_esc(c.get("answer") or "(no answer)")}</p></div>'
            "</div></details>"
        )
    more = len(ordered) - limit
    tail = (
        f"<p>{more} more cases moved; see the JSON report for all of them.</p>" if more > 0 else ""
    )
    return f'<div class="cases">{"".join(parts)}</div>{tail}'


def render_comparison_report(
    comparison: dict[str, Any],
    eval_results_baseline: dict[str, Any],
    eval_results_candidate: dict[str, Any],
    out_path: str | Path,
    title: str = "A vs B",
) -> Path:
    """Render a baseline-vs-candidate regression report page."""
    summary = comparison["summary"]
    rows = comparison.get("metrics", [])
    n_regressed = int(summary.get("n_regressed", 0))
    blocked = n_regressed > 0
    decision = (
        f"Blocked: {n_regressed} gated metric{'s' if n_regressed != 1 else ''} regressed"
        if blocked
        else "Passed: no gated metric regressed"
    )
    counts = (
        f"{_fmt(summary.get('n_improved', 0))} improved, "
        f"{_fmt(summary.get('n_inconclusive', 0))} inconclusive, "
        f"{_fmt(summary.get('n_neutral', 0))} neutral, "
        f"{_fmt(summary.get('n_ignored', 0))} reported but not gated"
    )
    labels = summary.get("labels", {})
    stats = comparison.get("statistics") or {}
    stats_text = (
        f"Paired tests at alpha {_fmt(stats.get('alpha'))}, {_fmt(stats.get('n_resamples'))} "
        f"bootstrap resamples (seed {_fmt(stats.get('seed'))}), tolerance rule below "
        f"{_fmt(stats.get('min_pairs'))} pairs."
        if stats.get("method") == "paired"
        else "Tolerance rule only (no paired tests)."
    )
    n_cases = eval_results_candidate.get("aggregates", {}).get("n_cases")
    warnings = comparison.get("warnings") or []
    warning_html = (
        "<h2>Warnings</h2><ul class='warnings'>"
        + "".join(f"<li>{_esc(w)}</li>" for w in warnings)
        + "</ul>"
        if warnings
        else ""
    )
    body = f"""
<header class="top">
  <h1>{_esc(title)}</h1>
  <p class="meta">Baseline {_esc(labels.get("baseline", "baseline"))}, candidate
  {_esc(labels.get("candidate", "candidate"))}, {_fmt(n_cases)} cases. {stats_text}</p>
  <div class="decision {"blocked" if blocked else "passed"}" role="status">
    <strong>{_esc(decision)}</strong>
    <p>{counts} ({_esc(summary.get("verdict", ""))}).</p>
  </div>
</header>
<main>
  <h2 id="metrics">Metrics</h2>
  <p>Each strip shows the paired change from baseline to candidate: the dot is the
  mean per-case change, the bar its 95% interval, the shaded band the tolerance.
  A change counts only when it leaves the band and the paired test supports it.</p>
  <div class="scroll" role="region" aria-label="Metric comparison" tabindex="0">
  <table>
    <thead><tr><th>Metric</th><th class="num">Baseline</th><th class="num">Candidate</th>
    <th class="num">Change</th><th>Paired change and 95% interval</th><th>Evidence</th>
    <th>Verdict</th></tr></thead>
    <tbody>{_metric_rows(rows)}</tbody>
  </table>
  </div>
  <h2 id="failures">Failure categories</h2>
  {_failure_table(eval_results_baseline, eval_results_candidate)}
  <h2 id="cases">Cases that moved</h2>
  <p>Cases whose own score moved beyond the tolerance on a gated metric, most affected first.</p>
  {_changed_cases(rows, eval_results_baseline, eval_results_candidate)}
  {warning_html}
</main>
"""
    return atomic_write_text(out_path, _page(title, body))


# --- single-run report --------------------------------------------------------


def _case_table(per_case: list[dict[str, Any]]) -> str:
    if not per_case:
        return "<p>No cases.</p>"
    cols = [c for c in _CASE_COLUMNS if any(c in row for row in per_case)]
    header = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(row.get(c))}</td>" for c in cols) + "</tr>" for row in per_case
    )
    return (
        '<div class="scroll" role="region" aria-label="Per-case results" tabindex="0">'
        f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def render_run_report(
    run_dir: str | Path,
    eval_results: dict[str, Any],
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Render a single-run report page."""
    run_dir = Path(run_dir)
    manifest: dict[str, Any] = {}
    mpath = run_dir / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    title = title or str(manifest.get("name", run_dir.name))
    agg = eval_results.get("aggregates", {})
    means = [(k.removeprefix("mean_"), v) for k, v in agg.items() if k.startswith("mean_")]
    metric_rows = "".join(
        f"<tr><td>{_esc(name)}</td><td class='num'>{_fmt(value)}</td>"
        f"<td class='num'>{_fmt(agg.get(f'n_{name}'))}</td>"
        f"<td class='num'>{_esc(' to '.join(_num(v) for v in agg.get(f'ci95_{name}') or []))}</td></tr>"
        for name, value in means
    )
    failures = agg.get("failure_counts", {})
    failure_text = ", ".join(f"{_esc(k)} {_fmt(v)}" for k, v in failures.items()) or "none"
    body = f"""
<header class="top">
  <h1>Run report: {_esc(title)}</h1>
  <p class="meta">{_fmt(agg.get("n_cases", len(eval_results.get("per_case", []))))} cases,
  success rate {_fmt(agg.get("success_rate"))}, token sources
  {_esc(", ".join(str(s) for s in agg.get("token_sources", [])) or "none")}.</p>
</header>
<main>
  <h2 id="metrics">Metrics</h2>
  <div class="scroll" role="region" aria-label="Aggregate metrics" tabindex="0">
  <table><thead><tr><th>Metric</th><th class="num">Mean</th><th class="num">Cases</th>
  <th class="num">95% interval</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
  <h2>Failures</h2>
  <p>{failure_text}</p>
  <h2>Per-case results</h2>
  {_case_table(eval_results.get("per_case", []))}
</main>
"""
    return atomic_write_text(out_path, _page(title, body))
