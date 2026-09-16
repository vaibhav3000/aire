#!/usr/bin/env python3
"""Validate the AIRE demo evaluation set against its demo corpus.

Checks performed on ``demo/eval_set.jsonl``:

1. Schema
   - every line parses as a JSON object with only known fields,
   - required fields (``case_id``, ``input``) are non-empty strings,
   - list fields hold strings, keywords are lowercase, boolean flags are
     booleans, the file holds exactly the expected number of cases, and
     ``case_id`` values are unique.
2. Answerable cases (``expect_abstention`` absent or false)
   - at least one supporting doc is listed and every listed doc exists
     in ``demo/corpus/``,
   - 3-5 expected keywords are given and every keyword occurs
     (case-insensitive substring) in the concatenated text of the
     supporting docs.
3. Abstention cases (``expect_abstention`` true)
   - ``expected_keywords`` and ``supporting_doc_ids`` are empty and
     ``must_cite`` is explicitly false, so no doc lookup applies.

The script prints a count breakdown and exits 0 when every check passes,
1 otherwise. Paths resolve relative to this file, so it runs from any
working directory:

    python demo/validate_eval_set.py        # from the aire/ root
    python aire/demo/validate_eval_set.py   # from the parent directory
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEMO_DIR = Path(__file__).resolve().parent
CORPUS_DIR = DEMO_DIR / "corpus"
EVAL_SET_PATH = DEMO_DIR / "eval_set.jsonl"

EXPECTED_CASE_COUNT = 26
MIN_KEYWORDS, MAX_KEYWORDS = 3, 5

REQUIRED_STRING_FIELDS = ("case_id", "input")
REQUIRED_LIST_FIELDS = ("expected_keywords", "supporting_doc_ids", "tags")
KNOWN_FIELDS = frozenset(
    REQUIRED_STRING_FIELDS
    + REQUIRED_LIST_FIELDS
    + ("expected_answer", "expected_tool", "expected_tool_args", "expect_abstention", "must_cite")
)


@dataclass(frozen=True)
class Case:
    """One parsed evaluation case plus its source line number."""

    line_number: int
    data: dict[str, Any]

    @property
    def case_id(self) -> str:
        value = self.data.get("case_id")
        return value if isinstance(value, str) else "<missing case_id>"

    @property
    def expects_abstention(self) -> bool:
        return self.data.get("expect_abstention") is True


def load_cases(path: Path, errors: list[str]) -> list[Case]:
    """Parse the JSONL file, recording parse failures instead of raising."""
    if not path.is_file():
        errors.append(f"missing evaluation set file: {path}")
        return []
    cases: list[Case] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON ({exc})")
            continue
        if not isinstance(payload, dict):
            errors.append(f"line {line_number}: expected a JSON object, got {type(payload).__name__}")
            continue
        cases.append(Case(line_number=line_number, data=payload))
    return cases


def load_corpus(errors: list[str]) -> dict[str, str]:
    """Map doc_id (filename without .md) to document text."""
    if not CORPUS_DIR.is_dir():
        errors.append(f"missing corpus directory: {CORPUS_DIR}")
        return {}
    corpus = {path.stem: path.read_text(encoding="utf-8") for path in sorted(CORPUS_DIR.glob("*.md"))}
    if not corpus:
        errors.append(f"no markdown documents found in {CORPUS_DIR}")
    return corpus


def check_schema(case: Case, errors: list[str]) -> None:
    """Structural checks that need no corpus access."""
    where = f"line {case.line_number} ({case.case_id})"
    for field_name in REQUIRED_STRING_FIELDS:
        value = case.data.get(field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{where}: field '{field_name}' must be a non-empty string")
    for field_name in REQUIRED_LIST_FIELDS:
        value = case.data.get(field_name)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            errors.append(f"{where}: field '{field_name}' must be a list of strings")
    unknown = sorted(set(case.data) - KNOWN_FIELDS)
    if unknown:
        errors.append(f"{where}: unknown field(s): {', '.join(unknown)}")
    expected_answer = case.data.get("expected_answer")
    if expected_answer is not None and not isinstance(expected_answer, str):
        errors.append(f"{where}: 'expected_answer' must be a string or null")
    for flag in ("expect_abstention", "must_cite"):
        if flag in case.data and not isinstance(case.data[flag], bool):
            errors.append(f"{where}: '{flag}' must be a boolean")
    keywords = case.data.get("expected_keywords")
    if isinstance(keywords, list):
        for keyword in keywords:
            if isinstance(keyword, str) and keyword != keyword.lower():
                errors.append(f"{where}: keyword {keyword!r} must be lowercase")


def check_answerable(case: Case, corpus: dict[str, str], errors: list[str]) -> None:
    """Every keyword must occur in the concatenated supporting docs."""
    where = f"line {case.line_number} ({case.case_id})"
    doc_ids = case.data.get("supporting_doc_ids")
    if not isinstance(doc_ids, list) or not doc_ids:
        errors.append(f"{where}: answerable case needs at least one supporting_doc_ids entry")
        return
    missing = [doc_id for doc_id in doc_ids if doc_id not in corpus]
    if missing:
        errors.append(f"{where}: unknown doc id(s): {', '.join(missing)}")
        return
    haystack = "\n".join(corpus[doc_id] for doc_id in doc_ids).lower()
    keywords = case.data.get("expected_keywords")
    if not isinstance(keywords, list):
        return  # schema check already reported the type problem
    if not MIN_KEYWORDS <= len(keywords) <= MAX_KEYWORDS:
        errors.append(f"{where}: expected {MIN_KEYWORDS}-{MAX_KEYWORDS} keywords, found {len(keywords)}")
    for keyword in keywords:
        if isinstance(keyword, str) and keyword.lower() not in haystack:
            errors.append(f"{where}: keyword {keyword!r} not found in {', '.join(doc_ids)}")


def check_abstention(case: Case, errors: list[str]) -> None:
    """Abstention cases must be keyword-free, doc-free, and citation-free."""
    where = f"line {case.line_number} ({case.case_id})"
    if case.data.get("expected_keywords") != []:
        errors.append(f"{where}: abstention case must set expected_keywords to []")
    if case.data.get("supporting_doc_ids") != []:
        errors.append(f"{where}: abstention case must set supporting_doc_ids to []")
    if case.data.get("must_cite") is not False:
        errors.append(f"{where}: abstention case must set must_cite to false")
    if "expected_answer" in case.data:
        errors.append(f"{where}: abstention case must not set expected_answer")


def validate(cases: list[Case], corpus: dict[str, str]) -> list[str]:
    """Run all checks; return the collected problems (empty means valid)."""
    errors: list[str] = []
    if len(cases) != EXPECTED_CASE_COUNT:
        errors.append(f"expected {EXPECTED_CASE_COUNT} cases, found {len(cases)}")

    seen_ids: set[str] = set()
    for case in cases:
        if case.data.get("case_id") in seen_ids:
            errors.append(f"line {case.line_number}: duplicate case_id {case.case_id!r}")
        if isinstance(case.data.get("case_id"), str):
            seen_ids.add(case.case_id)
        check_schema(case, errors)

    for case in cases:
        if case.expects_abstention:
            check_abstention(case, errors)
        else:
            check_answerable(case, corpus, errors)
    return errors


def print_breakdown(cases: list[Case], corpus: dict[str, str]) -> None:
    """Summarise how the cases are distributed across the corpus."""
    answerable = [case for case in cases if not case.expects_abstention]
    abstentions = [case for case in cases if case.expects_abstention]
    single_doc = [case for case in answerable if len(case.data.get("supporting_doc_ids", [])) == 1]
    multi_doc = [case for case in answerable if len(case.data.get("supporting_doc_ids", [])) > 1]
    referenced = {
        doc_id
        for case in answerable
        for doc_id in case.data.get("supporting_doc_ids", [])
        if doc_id in corpus
    }
    print(f"Corpus documents loaded : {len(corpus)}")
    print(f"Total cases             : {len(cases)}")
    print(f"  answerable, 1 doc     : {len(single_doc)}")
    print(f"  answerable, 2 docs    : {len(multi_doc)}")
    print(f"  abstention            : {len(abstentions)}")
    print(f"Distinct docs referenced: {len(referenced)}/{len(corpus)}")
    if corpus:
        unused = sorted(set(corpus) - referenced)
        print(f"Docs never referenced   : {', '.join(unused) if unused else 'none'}")


def main() -> int:
    errors: list[str] = []
    cases = load_cases(EVAL_SET_PATH, errors)
    corpus = load_corpus(errors)
    errors.extend(validate(cases, corpus))

    print_breakdown(cases, corpus)

    if errors:
        print(f"\nFAILED with {len(errors)} issue(s):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
