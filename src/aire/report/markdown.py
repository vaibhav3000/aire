"""The PR comment: a gate report rendered as GitHub-flavoured markdown.

The gate JSON comes from a pull request's own code, so it is untrusted
(ADR 0005). Rendering therefore:

- refuses a document that is not shaped like a gate report;
- prints numbers only when they are numbers;
- reduces every string to an allow-list of characters (letters, digits,
  space and a few punctuation marks) and wraps identifiers in code spans, so
  no mention, link, image, HTML or table break can be injected;
- caps the body well below GitHub's 65,536 character limit.

The first line is a hidden marker that identifies the comment to update.
"""

from __future__ import annotations

import re
from typing import Any

MARKER = "<!-- aire-pr-gate -->"
MAX_CHARS = 60_000
_UNSAFE = re.compile(r"[^A-Za-z0-9 _.,:;=+\-/()%#'\\]")
_ORDER = {
    "regressed": 0,
    "inconclusive": 1,
    "improved": 2,
    "not comparable": 3,
    "new": 4,
    "neutral": 5,
}
_TESTS = {
    "mcnemar_exact": "McNemar",
    "paired_bootstrap": "paired bootstrap",
    "point": "tolerance only",
}


def _safe(value: Any, limit: int = 120) -> str:
    return _UNSAFE.sub("_", str(value))[:limit]


def _code(value: Any) -> str:
    return f"`{_safe(value)}`"


def _num(value: Any, spec: str = ".4g") -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "-"
    return format(value, spec)


def _check(report: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(report, dict):
        raise ValueError("gate report must be a JSON object")
    summary, metrics = report.get("summary"), report.get("metrics")
    if not isinstance(summary, dict) or not isinstance(metrics, list):
        raise ValueError("gate report needs a summary object and a metrics list")
    for row in metrics:
        if not isinstance(row, dict) or not isinstance(row.get("metric"), str):
            raise ValueError("every metric row needs a string 'metric'")
        if not isinstance(row.get("verdict"), str):
            raise ValueError(f"metric {_safe(row.get('metric'))} has no verdict")
    return summary, metrics


def _evidence(row: dict[str, Any]) -> str:
    test = row.get("test")
    if test is None:
        return _safe(row.get("note") or "")
    label = _TESTS.get(str(test), _safe(test, 30))
    if test == "point":
        return f"{label}, {_num(row.get('n_pairs'), 'd')} pairs"
    p = row.get("p_value")
    p_text = "p < 0.001" if isinstance(p, int | float) and p < 0.001 else f"p = {_num(p, '.3g')}"
    return f"{label}, {p_text}"


def _verdict(row: dict[str, Any]) -> str:
    text = _safe(row["verdict"], 30)
    if row.get("direction_of_change") in ("better", "worse"):
        text += f" ({row['direction_of_change']})"
    if row.get("gated") is False:
        text += ", not gated"
    return f"**{text}**" if row["verdict"] == "regressed" and row.get("gated", True) else text


def _table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Metric | Baseline | Candidate | Change | Evidence | Verdict |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {_code(row['metric'])} | {_num(row.get('baseline'))} | {_num(row.get('candidate'))} "
            f"| {_num(row.get('delta'), '+.4g')} | {_evidence(row)} | {_verdict(row)} |"
        )
    return "\n".join(lines)


def _details_link(details_url: str | None) -> list[str]:
    if details_url and re.fullmatch(r"https://github\.com/[A-Za-z0-9_.\-/]+", details_url):
        return [f"[Full report, per-case details and the HTML diff]({details_url})"]
    return []


_SELF_GATE = {
    0: "passed",
    1: "**blocked**: a gated metric regressed against those results",
}


def _render_not_comparable(report: dict[str, Any], title: str, details_url: str | None) -> str:
    """The comment for a PR whose evaluation set differs from the base branch's."""
    if report.get("reason") != "dataset_changed" or not isinstance(report.get("dataset"), dict):
        raise ValueError("a not_comparable gate report needs reason dataset_changed and a dataset")
    dataset: dict[str, Any] = report["dataset"]
    raw_self = report.get("self_gate")
    self_exit = raw_self.get("exit_code") if isinstance(raw_self, dict) else None
    if isinstance(self_exit, int) and not isinstance(self_exit, bool):
        self_text = _SELF_GATE.get(self_exit, f"**failed to run** (exit {self_exit})")
    else:
        self_text = "not recorded"
    parts = [
        MARKER,
        f"### aire regression gate: {_safe(title, 80)}",
        "",
        "**Not comparable: this PR changes the evaluation set.**",
        "",
        f"{_code(dataset.get('candidate_path'))} has sha256 "
        f"{_code(dataset.get('baseline_sha256'))} on the base branch and "
        f"{_code(dataset.get('candidate_sha256'))} in this pull request, so its cases cannot "
        "be paired with the base branch's results and no metric was compared against them.",
        "",
        "The pull request's fresh run was gated against the results committed in the "
        f"pull request instead: {self_text}.",
        "",
    ]
    return "\n".join(parts + _details_link(details_url))


def render_gate_markdown(report: Any, *, title: str, details_url: str | None = None) -> str:
    if isinstance(report, dict) and report.get("status") == "not_comparable":
        return _render_not_comparable(report, title, details_url)
    summary, metrics = _check(report)
    rows = sorted(
        metrics,
        key=lambda r: (r.get("gated") is False, _ORDER.get(str(r["verdict"]), 9), str(r["metric"])),
    )
    moved = [r for r in rows if r.get("gated", True) and r["verdict"] != "neutral"]
    n_regressed = sum(1 for r in rows if r.get("gated", True) and r["verdict"] == "regressed")
    decision = (
        f"Blocked: {n_regressed} gated metric{'s' if n_regressed != 1 else ''} regressed"
        if n_regressed
        else "Passed: no gated metric regressed"
    )
    raw_labels = summary.get("labels")
    labels: dict[str, Any] = raw_labels if isinstance(raw_labels, dict) else {}
    raw_stats = report.get("statistics")
    stats: dict[str, Any] = raw_stats if isinstance(raw_stats, dict) else {}
    counts = ", ".join(
        f"{_num(summary.get(key), 'd') if isinstance(summary.get(key), int) else 0} {name}"
        for key, name in (
            ("n_improved", "improved"),
            ("n_inconclusive", "inconclusive"),
            ("n_neutral", "neutral"),
        )
    )
    parts = [
        MARKER,
        f"### aire regression gate: {_safe(title, 80)}",
        "",
        f"**{decision}**",
        "",
        f"{counts}. Baseline {_code(labels.get('baseline', 'baseline'))}, "
        f"candidate {_code(labels.get('candidate', 'candidate'))}"
        + (
            f"; paired tests at alpha {_num(stats.get('alpha'))}."
            if stats.get("method") == "paired"
            else "; tolerance rule only."
        ),
        "",
    ]
    if moved:
        parts += [_table(moved), ""]
    details = ["<details><summary>All metrics</summary>", "", _table(rows), "", "</details>", ""]
    raw_warnings = report.get("warnings")
    warnings: list[Any] = raw_warnings if isinstance(raw_warnings, list) else []
    notes = [f"- {_safe(w, 300)}" for w in warnings[:10]]
    if notes:
        notes = ["Warnings:", *notes, ""]
    link = _details_link(details_url)
    body = "\n".join(parts + details + notes + link)
    if len(body) > MAX_CHARS:  # drop the full table first, then cut
        body = "\n".join(parts + notes + link)
    return body[:MAX_CHARS]
