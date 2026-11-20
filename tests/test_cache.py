"""Response cache: exact-request keys, atomic entries, corrupt entries are misses."""

from pathlib import Path

import pytest

from aire.providers.cache import ResponseCache, request_key
from aire.providers.types import ChatRequest, ChatResponse, Message, TokenUsage


def _request(**overrides: object) -> ChatRequest:
    fields: dict[str, object] = {"model": "m", "messages": (Message("user", "What is 2+2?"),)}
    fields.update(overrides)
    return ChatRequest(**fields)  # type: ignore[arg-type]


def _response() -> ChatResponse:
    return ChatResponse(
        text="4",
        usage=TokenUsage(input_tokens=5, output_tokens=1),
        response_model="m",
        finish_reason="stop",
    )


def test_key_changes_with_everything_that_changes_the_output() -> None:
    base = request_key("openai", _request())
    assert request_key("openai", _request()) == base
    assert request_key("anthropic", _request()) != base
    assert request_key("openai", _request(model="m2")) != base
    assert request_key("openai", _request(temperature=0.7)) != base
    assert request_key("openai", _request(max_output_tokens=10)) != base
    assert request_key("openai", _request(seed=1)) != base
    assert request_key("openai", _request(messages=(Message("user", "What is 2+3?"),))) != base
    assert request_key("openai", _request(json_schema={"type": "object"})) != base


def test_key_ignores_the_order_of_schema_keys() -> None:
    a = _request(json_schema={"type": "object", "required": ["x"]})
    b = _request(json_schema={"required": ["x"], "type": "object"})
    assert request_key("openai", a) == request_key("openai", b)


def test_entries_round_trip_with_their_original_latency(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    key = request_key("openai", _request())
    assert cache.get(key) is None
    cache.put(key, _response(), latency_ms=812.5)
    entry = cache.get(key)
    assert entry is not None
    assert entry.response == _response()
    assert entry.latency_ms == 812.5
    assert (tmp_path / key[:2] / f"{key}.json").is_file()


def test_a_corrupt_entry_is_a_miss_and_a_warning(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    key = request_key("openai", _request())
    path = tmp_path / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="unreadable cache entry"):
        assert cache.get(key) is None
