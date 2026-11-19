"""The JSON Schema subset aire validates: tool arguments and structured outputs."""

import pytest

from aire.jsonschema import SchemaError, validate

TOOL = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "minLength": 1},
        "start_line": {"type": "integer", "minimum": 1},
        "mode": {"enum": ["r", "w"]},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    },
    "required": ["path"],
    "additionalProperties": False,
}


def test_a_valid_instance_has_no_problems() -> None:
    assert validate({"path": "a.py", "start_line": 3, "mode": "r", "tags": ["x"]}, TOOL) == []


@pytest.mark.parametrize(
    ("instance", "fragment"),
    [
        ({}, "missing required property 'path'"),
        ({"path": ""}, "shorter than 1"),
        ({"path": "a", "start_line": 0}, "below the minimum 1"),
        ({"path": "a", "start_line": True}, "expected integer"),
        ({"path": "a", "start_line": 1.5}, "expected integer"),
        ({"path": "a", "mode": "x"}, "not one of"),
        ({"path": "a", "tags": ["a", "b", "c"]}, "more than 2 items"),
        ({"path": "a", "tags": [1]}, "$.tags[0]: expected string"),
        ({"path": "a", "extra": 1}, "unexpected property 'extra'"),
        ([], "expected object"),
    ],
)
def test_each_violation_is_reported_with_its_location(instance: object, fragment: str) -> None:
    problems = validate(instance, TOOL)
    assert any(fragment in p for p in problems), problems


def test_numbers_accept_integers_but_booleans_are_never_numbers() -> None:
    assert validate(3, {"type": "number"}) == []
    assert validate(True, {"type": "number"}) != []
    assert validate(None, {"type": ["string", "null"]}) == []


def test_const_any_of_and_nested_additional_properties() -> None:
    schema = {
        "anyOf": [{"const": "tie"}, {"type": "object", "additionalProperties": {"type": "number"}}]
    }
    assert validate("tie", schema) == []
    assert validate({"a": 1.0}, schema) == []
    assert validate({"a": "x"}, schema) != []


def test_unsupported_keywords_are_refused_rather_than_ignored() -> None:
    with pytest.raises(SchemaError, match="pattern"):
        validate("abc", {"type": "string", "pattern": "^a"})
