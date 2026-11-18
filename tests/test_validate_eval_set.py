"""demo/validate_eval_set.py must use the engine's keyword matcher."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_eval_set", ROOT / "demo" / "validate_eval_set.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules, so register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _answerable_case(validator: ModuleType, keywords: list[str]) -> object:
    return validator.Case(
        line_number=1,
        data={
            "case_id": "c1",
            "input": "Is there an NDA?",
            "expected_keywords": keywords,
            "supporting_doc_ids": ["doc_a"],
            "tags": [],
        },
    )


def test_keyword_found_only_inside_a_longer_word_is_rejected() -> None:
    validator = _load_validator()
    corpus = {"doc_a": "The standard plan includes 24-hour support and a 95% target."}
    errors: list[str] = []
    case = _answerable_case(validator, ["nda", "4-hour", "5%"])
    validator.check_answerable(case, corpus, errors)
    assert len(errors) == 3, errors


def test_keyword_present_as_whole_tokens_is_accepted() -> None:
    validator = _load_validator()
    corpus = {"doc_a": "Reports are shared under NDA, with a 4-hour target and a 5% credit."}
    errors: list[str] = []
    case = _answerable_case(validator, ["nda", "4-hour", "5%"])
    validator.check_answerable(case, corpus, errors)
    assert errors == []
