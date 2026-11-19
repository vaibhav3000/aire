"""Structured-output parsing shared by every adapter.

The SDKs disagree on what happens when a model's JSON does not match the
requested schema (Gemini leaves parsed=None silently, Anthropic's helper
raises), so aire parses and validates the text itself and treats a mismatch
as a malformed response that retrying the same request will not fix.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..jsonschema import validate
from .errors import ErrorKind, ProviderError

_FENCE = "```"


def parse_structured(text: str, schema: Mapping[str, Any], *, provider: str = "") -> Any:
    """The JSON value in `text`, validated against `schema`."""
    candidate = text.strip()
    if candidate.startswith(_FENCE):  # tolerate a ```json fenced block from local models
        candidate = candidate.strip("`")
        candidate = candidate.removeprefix("json").strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"structured output is not valid JSON: {exc}",
            kind=ErrorKind.MALFORMED,
            provider=provider,
        ) from exc
    problems = validate(value, schema)
    if problems:
        raise ProviderError(
            f"structured output does not match the schema: {'; '.join(problems[:5])}",
            kind=ErrorKind.MALFORMED,
            provider=provider,
        )
    return value
