#!/usr/bin/env python
"""Render the measured numbers in the docs from committed files.

    python scripts/sync_docs.py           # rewrite every generated region
    python scripts/sync_docs.py --check   # exit 1 when a region is out of date

README.md, docs/design_faq.md and the technical report hold their measured
numbers inside marked regions:

    <!-- aire:begin NAME -->      (markdown)      % aire:begin NAME   (LaTeX)
    ...generated text...
    <!-- aire:end NAME -->                        % aire:end NAME

The text of region NAME is docs/templates/NAME.md (or .tex) with each
placeholder ``{{key}}`` or ``{{key|format}}`` replaced by a value read from the
committed results, the evaluation set or the corpus (see ``collect_facts``).
Nothing in a region is typed by hand; edit the template and rerun.
tests/test_docs_numbers.py runs ``--check`` and also fails on any measured
number written outside a region.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEMPLATES = ROOT / "docs" / "templates"
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs" / "design_faq.md",
    ROOT / "reports" / "AIRE_AI_Reliability_Evaluation_Engine.tex",
)
VERSIONS = {"v1": "v1_baseline", "v2": "v2_improved", "v3": "v3_fluent_regression"}
COMPARISONS = {
    "v2": "v2_improved_vs_v1_baseline.json",
    "v3": "v3_fluent_regression_vs_v1_baseline.json",
    "llm_hist": "llm_vs_deterministic.json",
}
LIVE_RUN = RESULTS / "llm_gemini-2.5-flash"

_MARKER = re.compile(
    r"^(?:<!-- aire:(?P<md>begin|end) (?P<md_name>[\w-]+) -->"
    r"|% aire:(?P<tex>begin|end) (?P<tex_name>[\w-]+))\s*$"
)
_PLACEHOLDER = re.compile(r"\{\{\s*([\w.-]+)\s*(?:\|\s*([^}]*?)\s*)?\}\}")
_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]


class DocsError(Exception):
    """A document, template or fact is missing or malformed."""


# Facts


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten(value: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(item, f"{prefix}.{key}", out)
    else:
        out[prefix] = value


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield json.loads(line)


def _dataset_facts(facts: dict[str, Any]) -> None:
    cases = list(_read_jsonl(ROOT / "demo" / "eval_set.jsonl"))
    unanswerable = [c for c in cases if c.get("expect_abstention")]
    facts["ds.n_cases"] = len(cases)
    facts["ds.n_unanswerable"] = len(unanswerable)
    facts["ds.n_answerable"] = len(cases) - len(unanswerable)
    facts["ds.n_two_doc"] = sum(len(c.get("supporting_doc_ids", [])) >= 2 for c in cases)
    facts["ds.n_docs"] = sum(1 for p in (ROOT / "demo" / "corpus").iterdir() if p.is_file())


def _comparison_facts(facts: dict[str, Any]) -> None:
    for key, name in COMPARISONS.items():
        data = _load(RESULTS / name)
        _flatten(data["summary"], f"cmp.{key}.summary", facts)
        for row in data["metrics"]:
            _flatten(row, f"cmp.{key}.{row['metric']}", facts)
        _flatten(data.get("failure_count_deltas", {}), f"cmp.{key}.fail", facts)


def _historical_llm_facts(facts: dict[str, Any]) -> None:
    data = _load(RESULTS / "llm_eval_results.json")
    _flatten(data["aggregates"], "llm_hist", facts)
    for key in ("total_api_prompt_tokens", "total_api_completion_tokens", "total_api_calls"):
        facts[f"llm_hist.{key}"] = data[key]
    rows = data["per_case"]
    facts["llm_hist.n_grounded"] = sum(r.get("groundedness") is not None for r in rows)
    facts["llm_hist.n_unanswerable"] = sum(r.get("abstention_correct") is not None for r in rows)
    facts["llm_hist.mean_latency_s"] = data["aggregates"]["mean_latency_ms"] / 1000


def _live_run_facts(facts: dict[str, Any]) -> None:
    manifest = _load(LIVE_RUN / "manifest.json")
    traces = list(_read_jsonl(LIVE_RUN / "traces.jsonl"))
    answered = [t for t in traces if not t.get("error")]
    usage = [t.get("usage") or {} for t in answered]
    created = datetime.fromtimestamp(manifest["created_at"], tz=UTC)
    facts["live.date"] = created.strftime("%Y-%m-%d")
    facts["live.model"] = manifest["system_config"]["generator"].split(":", 1)[-1]
    facts["live.n_cases"] = manifest["n_cases"]
    facts["live.n_answered"] = len(answered)
    facts["live.n_unanswered"] = manifest["n_cases"] - len(answered)
    facts["live.complete"] = len(answered) == manifest["n_cases"]
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
        facts[f"live.{key}"] = sum(u.get(key) or 0 for u in usage)


def _benchmark_facts(facts: dict[str, Any]) -> None:
    data = _load(RESULTS / "runner_benchmark.json")
    facts["bench.recorded_at_utc"] = data["recorded_at_utc"]
    facts["bench.n_cases"] = data["n_cases"]
    _flatten(data["server"], "bench.server", facts)
    _flatten(data["machine"], "bench.machine", facts)
    for run in data["runs"]:
        _flatten(run, f"bench.{run['label']}", facts)
    sequential = facts["bench.sequential.wall_time_s"]
    for run in data["runs"]:
        facts[f"bench.{run['label']}.speedup"] = sequential / run["wall_time_s"]


def collect_facts() -> dict[str, Any]:
    """Every value a template may use, keyed by a dotted name."""
    facts: dict[str, Any] = {}
    categories: set[str] = set()
    for key, name in VERSIONS.items():
        aggregates = _load(RESULTS / name / "eval_results.json")["aggregates"]
        _flatten(aggregates, key, facts)
        categories.update(aggregates["failure_counts"])
        # Correct refusals as a count, for prose such as "2 of the 20".
        facts[f"{key}.n_abstained"] = round(
            aggregates["mean_abstention_correct"] * aggregates["n_abstention_correct"]
        )
    # A category no case fell into is absent from failure_counts; it is 0.
    for key in VERSIONS:
        for category in categories:
            facts.setdefault(f"{key}.failure_counts.{category}", 0)
    _flatten(_load(RESULTS / "retrieval_ablation.json"), "abl", facts)
    _flatten(_load(RESULTS / "judge_calibration.json"), "calib", facts)
    _dataset_facts(facts)
    _comparison_facts(facts)
    _historical_llm_facts(facts)
    _live_run_facts(facts)
    _benchmark_facts(facts)
    return facts


# Rendering


def _number_word(value: Any, capital: bool) -> str:
    if not isinstance(value, int) or not 0 <= value < len(_WORDS):
        raise DocsError(f"no word for {value!r}")
    word = _WORDS[value]
    return word.capitalize() if capital else word


FORMATS: dict[str, Callable[[Any], str]] = {
    "word": lambda v: _number_word(v, capital=False),
    "Word": lambda v: _number_word(v, capital=True),
    "signed": lambda v: f"{v:+d}",
    # LaTeX thousands separator without the extra space after the comma.
    "tex2": lambda v: f"{v:,.2f}".replace(",", "{,}"),
    "thousands": lambda v: f"{v:,}",
}


def format_value(value: Any, fmt: str | None) -> str:
    if fmt is None:
        return str(value)
    if fmt in FORMATS:
        return FORMATS[fmt](value)
    if value is None or isinstance(value, bool | str):
        raise DocsError(f"format {fmt!r} needs a number, got {value!r}")
    return format(value, fmt)


def render(template: str, facts: Mapping[str, Any], source: str) -> str:
    def substitute(match: re.Match[str]) -> str:
        key, fmt = match.group(1), match.group(2)
        if key not in facts:
            raise DocsError(f"{source}: unknown fact {key!r}")
        return format_value(facts[key], fmt)

    return _PLACEHOLDER.sub(substitute, template)


def _template(name: str, suffix: str) -> Path:
    path = TEMPLATES / f"{name}{suffix}"
    if not path.is_file():
        raise DocsError(f"region {name!r} has no template {path.relative_to(ROOT)}")
    return path


def regenerate(text: str, suffix: str, facts: Mapping[str, Any], source: str) -> str:
    """Return ``text`` with every region re-rendered from its template."""
    lines = text.split("\n")
    out: list[str] = []
    open_name: str | None = None
    for number, line in enumerate(lines, start=1):
        match = _MARKER.match(line)
        if match is None:
            if open_name is None:
                out.append(line)
            continue
        kind = match.group("md") or match.group("tex")
        name = match.group("md_name") or match.group("tex_name")
        if kind == "begin":
            if open_name is not None:
                raise DocsError(f"{source}:{number}: region {name!r} opens inside {open_name!r}")
            open_name = name
            out.append(line)
            body = _template(name, suffix).read_text(encoding="utf-8").replace("\r\n", "\n")
            out.extend(render(body.rstrip("\n"), facts, f"{source} [{name}]").split("\n"))
        else:
            if name != open_name:
                raise DocsError(f"{source}:{number}: end of {name!r} but {open_name!r} is open")
            open_name = None
            out.append(line)
    if open_name is not None:
        raise DocsError(f"{source}: region {open_name!r} is never closed")
    return "\n".join(out)


def sync(check: bool) -> list[str]:
    """Rewrite (or, with ``check``, compare) every document; return the stale ones."""
    facts = collect_facts()
    stale: list[str] = []
    for path in DOCUMENTS:
        current = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        suffix = ".tex" if path.suffix == ".tex" else ".md"
        rel = path.relative_to(ROOT).as_posix()
        fresh = regenerate(current, suffix, facts, rel)
        if fresh != current:
            stale.append(rel)
            if not check:
                path.write_text(fresh, encoding="utf-8", newline="\n")
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the measured numbers in the docs.")
    parser.add_argument(
        "--check", action="store_true", help="change nothing; exit 1 if a region is stale"
    )
    args = parser.parse_args(argv)
    try:
        stale = sync(check=args.check)
    except DocsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.check:
        for rel in stale:
            print(f"stale: {rel} (run python scripts/sync_docs.py)", file=sys.stderr)
        return 1 if stale else 0
    for rel in stale:
        print(f"updated {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
