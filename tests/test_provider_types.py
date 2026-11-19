"""Provider-neutral request and response types and HTTP error classification."""

import email.utils
import time

import pytest

from aire.providers.errors import (
    ErrorKind,
    ProviderError,
    classify_http_status,
    retry_after_seconds,
)
from aire.providers.types import ChatRequest, ChatResponse, Message, TokenUsage


def _request(**overrides: object) -> ChatRequest:
    fields: dict[str, object] = {
        "model": "m",
        "messages": (Message("system", "Be brief."), Message("user", "Hello there")),
    }
    fields.update(overrides)
    return ChatRequest(**fields)  # type: ignore[arg-type]


def test_estimated_tokens_counts_prompt_characters_and_the_output_cap() -> None:
    request = _request(max_output_tokens=100)
    prompt_chars = len("Be brief.") + len("Hello there")
    assert request.estimated_tokens() == prompt_chars // 4 + 100


def test_request_rejects_unknown_roles_and_empty_messages() -> None:
    with pytest.raises(ValueError, match="role"):
        Message("robot", "hi")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="message"):
        _request(messages=())


def test_response_round_trips_through_a_dict() -> None:
    response = ChatResponse(
        text='{"a": 1}',
        usage=TokenUsage(input_tokens=10, output_tokens=3, cache_read_tokens=4),
        response_model="m-001",
        finish_reason="stop",
        response_id="r1",
        parsed={"a": 1},
    )
    assert ChatResponse.from_dict(response.to_dict()) == response


def test_quota_exhaustion_is_not_a_rate_limit() -> None:
    error = classify_http_status(429, '{"error": {"code": "insufficient_quota"}}', {})
    assert error.kind is ErrorKind.QUOTA_EXHAUSTED
    gemini = classify_http_status(
        429, '{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}', {}
    )
    assert gemini.kind is ErrorKind.QUOTA_EXHAUSTED


def test_a_plain_429_is_retryable_with_the_server_wait() -> None:
    error = classify_http_status(429, "slow down", {"retry-after": "7"})
    assert error.kind is ErrorKind.RATE_LIMITED
    assert error.retry_after_s == 7.0
    assert error.retryable


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (408, ErrorKind.TIMEOUT),
        (500, ErrorKind.TRANSIENT),
        (503, ErrorKind.TRANSIENT),
        (529, ErrorKind.TRANSIENT),
        (400, ErrorKind.CLIENT),
        (401, ErrorKind.CLIENT),
        (404, ErrorKind.CLIENT),
        (422, ErrorKind.CLIENT),
    ],
)
def test_statuses_map_to_kinds(status: int, kind: ErrorKind) -> None:
    error = classify_http_status(status, "", {})
    assert error.kind is kind
    assert error.status == status


def test_client_errors_are_not_retryable() -> None:
    assert not ProviderError("bad", kind=ErrorKind.CLIENT).retryable
    assert not ProviderError("bad", kind=ErrorKind.MALFORMED).retryable
    assert not ProviderError("gone", kind=ErrorKind.QUOTA_EXHAUSTED).retryable


def test_retry_after_reads_milliseconds_dates_and_gemini_bodies() -> None:
    assert retry_after_seconds({"retry-after-ms": "1500"}, "") == 1.5
    assert retry_after_seconds({"Retry-After": "2"}, "") == 2.0
    future = email.utils.formatdate(time.time() + 30, usegmt=True)
    wait = retry_after_seconds({"Retry-After": future}, "")
    assert wait is not None and 25.0 <= wait <= 31.0
    assert retry_after_seconds({}, '{"retryDelay": "12s"}') == 12.0
    assert retry_after_seconds(None, "no hint") is None
