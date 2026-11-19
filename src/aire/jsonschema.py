"""A small JSON Schema validator for tool arguments and structured outputs.

Supported keywords: type (one or a list), properties, required,
additionalProperties (bool or schema), items, enum, const, anyOf, minimum,
maximum, exclusiveMinimum, exclusiveMaximum, minLength, maxLength, minItems,
maxItems; title, description, default and $schema are annotations. Any other
keyword raises SchemaError, so a schema that relies on an unsupported rule
fails loudly instead of being half checked.

As in JSON Schema, an integer is a number, and a boolean is neither.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ANNOTATIONS = frozenset({"title", "description", "default", "$schema", "examples"})
_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "anyOf",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    }
)


class SchemaError(ValueError):
    """The schema itself uses something this validator does not support."""


def _is_type(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    raise SchemaError(f"unknown type {name!r}")


def validate(instance: Any, schema: Mapping[str, Any]) -> list[str]:
    """Problems with `instance` against `schema`, as '$.path: message' strings."""
    problems: list[str] = []
    _check(instance, schema, "$", problems)
    return problems


def _check(value: Any, schema: Mapping[str, Any], where: str, out: list[str]) -> None:
    unknown = set(schema) - _KEYWORDS - _ANNOTATIONS
    if unknown:
        raise SchemaError(f"unsupported schema keyword(s) at {where}: {', '.join(sorted(unknown))}")

    if "type" in schema:
        wanted = schema["type"]
        names = [wanted] if isinstance(wanted, str) else list(wanted)
        if not any(_is_type(value, n) for n in names):
            out.append(f"{where}: expected {' or '.join(names)}, got {type(value).__name__}")
            return
    if "const" in schema and value != schema["const"]:
        out.append(f"{where}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{where}: {value!r} is not one of {schema['enum']!r}")
    if "anyOf" in schema:
        branches = [validate(value, branch) for branch in schema["anyOf"]]
        if all(branches):
            out.append(f"{where}: matches none of the anyOf alternatives")

    if _is_type(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(f"{where}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            out.append(f"{where}: {value} is above the maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            out.append(f"{where}: {value} is not above {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            out.append(f"{where}: {value} is not below {schema['exclusiveMaximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append(f"{where}: string shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append(f"{where}: string longer than {schema['maxLength']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            out.append(f"{where}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append(f"{where}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                _check(item, schema["items"], f"{where}[{i}]", out)
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                out.append(f"{where}: missing required property {name!r}")
        properties: Mapping[str, Any] = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            path = f"{where}.{name}"
            if name in properties:
                _check(item, properties[name], path, out)
            elif extra is False:
                out.append(f"{where}: unexpected property {name!r}")
            elif isinstance(extra, Mapping):
                _check(item, extra, path, out)
