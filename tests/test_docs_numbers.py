"""The measured numbers in the docs are generated from committed results.

scripts/sync_docs.py renders marked regions of README.md, docs/design_faq.md
and the technical report from templates; these tests fail the build when a
region no longer matches what the committed results say.
"""

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sync_docs", ROOT / "scripts" / "sync_docs.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sync_docs = _load_script()


def test_committed_docs_match_the_committed_results() -> None:
    stale = sync_docs.sync(check=True)
    assert stale == [], f"run python scripts/sync_docs.py; stale: {stale}"


def test_every_template_is_used_by_exactly_one_region() -> None:
    used: list[str] = []
    for path in sync_docs.DOCUMENTS:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = sync_docs._MARKER.match(line)
            if match and (match.group("md") or match.group("tex")) == "begin":
                used.append(match.group("md_name") or match.group("tex_name"))
    templates = sorted(p.stem for p in sync_docs.TEMPLATES.iterdir() if p.is_file())
    assert sorted(used) == templates


def test_a_changed_result_makes_its_region_stale() -> None:
    facts = sync_docs.collect_facts()
    text = (
        "before\n<!-- aire:begin readme-ablation -->\n"
        + sync_docs.render(
            (sync_docs.TEMPLATES / "readme-ablation.md").read_text(encoding="utf-8").rstrip("\n"),
            facts,
            "test",
        )
        + "\n<!-- aire:end readme-ablation -->\nafter"
    )
    assert sync_docs.regenerate(text, ".md", facts, "test") == text
    moved = dict(facts, **{"abl.retrievers.idf.recall_at_3": 0.5})
    fresh = sync_docs.regenerate(text, ".md", moved, "test")
    assert fresh != text
    assert "recall@3 0.8731" in fresh and ", 0.5000 and" in fresh
    assert fresh.startswith("before\n") and fresh.endswith("\nafter")


def test_text_typed_inside_a_region_is_replaced() -> None:
    facts = sync_docs.collect_facts()
    text = (
        "<!-- aire:begin readme-demo-intro -->\nhand-typed 151 cases\n"
        "<!-- aire:end readme-demo-intro -->"
    )
    fresh = sync_docs.regenerate(text, ".md", facts, "test")
    assert "hand-typed" not in fresh
    assert f"({facts['ds.n_cases']} cases:" in fresh


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("<!-- aire:begin readme-ablation -->\nno end", "never closed"),
        ("<!-- aire:end readme-ablation -->", "is open"),
        ("<!-- aire:begin no-such-region -->\n<!-- aire:end no-such-region -->", "no template"),
    ],
)
def test_malformed_regions_are_errors(text: str, message: str) -> None:
    with pytest.raises(sync_docs.DocsError, match=message):
        sync_docs.regenerate(text, ".md", sync_docs.collect_facts(), "test")


def test_an_unknown_fact_is_an_error() -> None:
    with pytest.raises(sync_docs.DocsError, match="unknown fact 'nope'"):
        sync_docs.render("{{nope|.4f}}", {}, "test")


@pytest.mark.parametrize(
    ("value", "fmt", "expected"),
    [
        (0.71538, ".4f", "0.7154"),
        (0.070312, ".1g", "0.07"),
        (109, "signed", "+109"),
        (-24, "signed", "-24"),
        (17, "Word", "Seventeen"),
        (4, "word", "four"),
        (15319.8296, "tex2", "15{,}319.83"),
        (150, None, "150"),
    ],
)
def test_formats(value: object, fmt: str | None, expected: str) -> None:
    assert sync_docs.format_value(value, fmt) == expected


# A measured-looking number: a decimal, a signed integer, "N of M", or a count
# of cases, documents, questions and similar.
_NUMBER = re.compile(
    r"(?<![\w.:/@-])("
    r"\d+\.\d+(?:\.\d+)*"
    r"|\d+ of (?:the )?\d+"
    r"|[+-]\d+(?:\.\d+)?"
    r"|\d+(?:,\d{3})*[- ](?:case|cases|document|documents|questions|answerable|unanswerable"
    r"|calls|requests|tokens|items|labels|traces|metrics)\b"
    r")(?![\w/])"
)

# Numbers allowed outside generated regions, and why. Anything else must come
# from a template so that it is rendered from committed results.
ALLOWED = {
    "3.11": "supported Python version",
    "3.13": "supported Python version",
    "3.14": "supported Python version",
    "0.02": "default gate tolerance, checked against RegressionThresholds below",
    "0.5": "default thresholds, checked against EngineConfig below",
    "0.0": "the demo's zero token price",
    "0.10": "illustrative arithmetic in the tolerance answer of the FAQ",
    "0.085": "illustrative arithmetic in the tolerance answer of the FAQ",
    "-15": "illustrative arithmetic in the tolerance answer of the FAQ",
    "1.5": "illustrative arithmetic in the tolerance answer of the FAQ",
    "0.34": "LaTeX column width",
    "0.58": "LaTeX column width",
    "2.5": "the model name Gemini 2.5 Flash",
    "127.0.0.1": "the loopback address of the fake server",
    "14 of 26": "historical watchdog incident; no committed run records it",
    "26 cases": "historical watchdog incident; no committed run records it",
}


def _numbers_outside_regions(path: Path) -> list[str]:
    found: list[str] = []
    inside = fenced = False
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        marker = sync_docs._MARKER.match(line)
        if marker:
            inside = (marker.group("md") or marker.group("tex")) == "begin"
            continue
        if line.startswith("```"):
            fenced = not fenced
            continue
        if inside or fenced or line.lstrip().startswith("%"):
            continue
        line = re.sub(r"https?://\S+", "", line)
        for token in _NUMBER.findall(line):
            if token not in ALLOWED:
                found.append(f"{path.name}:{number}: {token}")
    return found


@pytest.mark.parametrize("path", sync_docs.DOCUMENTS, ids=lambda p: p.name)
def test_no_measured_number_is_typed_outside_a_region(path: Path) -> None:
    assert _numbers_outside_regions(path) == []


def test_allowed_constants_match_the_code() -> None:
    from aire.evaluator.engine import EngineConfig
    from aire.regression.compare import RegressionThresholds

    assert RegressionThresholds().default == float("0.02")
    config = EngineConfig()
    assert config.groundedness_threshold == config.keyword_threshold == float("0.5")


def test_the_scanner_flags_a_typed_number(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text(
        "recall rose to 0.8615 on 130 cases, 2 of 20 refused, ungrounded +109\n"
        "<!-- aire:begin x -->\n0.9999 is fine here\n<!-- aire:end x -->\n"
        "```\naire gate --threshold 0.3\n```\n",
        encoding="utf-8",
    )
    found = [entry.split(": ", 1)[1] for entry in _numbers_outside_regions(doc)]
    assert found == ["0.8615", "130 cases", "2 of 20", "+109"]
