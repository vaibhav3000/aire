"""The measured numbers in the docs are generated from committed results.

scripts/sync_docs.py renders marked regions of README.md, docs/design_faq.md
and the technical report from templates; these tests fail the build when a
region no longer matches what the committed results say.
"""

import importlib.util
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
