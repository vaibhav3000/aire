"""ModelClient: retries, server waits, the quota circuit, caching and single flight.

The adapter is a ScriptedChatModel and sleeping is recorded, never real.
"""

import asyncio
import random
from pathlib import Path

import pytest

from aire.providers.cache import ResponseCache
from aire.providers.client import ModelCallError, ModelClient, RetryPolicy
from aire.providers.errors import ErrorKind, ProviderError
from aire.providers.fake import ScriptedChatModel
from aire.providers.ratelimit import AdaptiveRateLimiter
from aire.providers.types import ChatRequest, ChatResponse, Message, TokenUsage

REQUEST = ChatRequest(model="m", messages=(Message("user", "hello"),), max_output_tokens=10)


def _ok(text: str = "hi") -> ChatResponse:
    return ChatResponse(
        text=text,
        usage=TokenUsage(input_tokens=3, output_tokens=2),
        response_model="m",
        finish_reason="stop",
    )


def _err(kind: ErrorKind, retry_after: float | None = None, status: int = 429) -> ProviderError:
    return ProviderError(f"{kind}", kind=kind, status=status, retry_after_s=retry_after)


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _client(model: ScriptedChatModel, **kwargs: object) -> tuple[ModelClient, SleepRecorder]:
    sleep = SleepRecorder()
    client = ModelClient(model, sleep=sleep, rng=random.Random(7), **kwargs)  # type: ignore[arg-type]
    return client, sleep


def test_rate_limits_are_retried_with_the_server_wait_then_jittered_backoff() -> None:
    model = ScriptedChatModel(
        [_err(ErrorKind.RATE_LIMITED, 2.0), _err(ErrorKind.RATE_LIMITED), _ok()]
    )
    client, sleep = _client(model, retry=RetryPolicy(max_attempts=4, base_delay_s=0.5))
    completion = asyncio.run(client.complete(REQUEST))
    assert completion.response.text == "hi"
    assert [a.error_kind for a in completion.attempts] == ["rate_limited", "rate_limited", None]
    assert sleep.calls[0] == 2.0  # the server's own request
    assert 0.0 <= sleep.calls[1] <= 1.0  # full jitter: uniform(0, 0.5 * 2**1)
    assert completion.attempts[1].backoff_s == 2.0
    assert completion.waited_s == pytest.approx(sum(sleep.calls))
    assert len(model.requests) == 3


def test_a_server_wait_above_the_cap_fails_instead_of_sleeping() -> None:
    model = ScriptedChatModel([_err(ErrorKind.RATE_LIMITED, 120.0)])
    client, sleep = _client(model, retry=RetryPolicy(max_delay_s=30.0))
    with pytest.raises(ModelCallError, match="above the 30s cap") as info:
        asyncio.run(client.complete(REQUEST))
    assert sleep.calls == []
    assert len(info.value.attempts) == 1


def test_client_and_malformed_errors_are_not_retried() -> None:
    for kind in (ErrorKind.CLIENT, ErrorKind.MALFORMED):
        model = ScriptedChatModel([_err(kind, status=400), _ok()])
        client, _ = _client(model)
        with pytest.raises(ModelCallError) as info:
            asyncio.run(client.complete(REQUEST))
        assert info.value.kind is kind
        assert len(model.requests) == 1


def test_attempts_stop_at_the_policy_limit() -> None:
    model = ScriptedChatModel([_err(ErrorKind.TRANSIENT, status=503)] * 5)
    client, sleep = _client(model, retry=RetryPolicy(max_attempts=3))
    with pytest.raises(ModelCallError) as info:
        asyncio.run(client.complete(REQUEST))
    assert len(info.value.attempts) == 3
    assert len(sleep.calls) == 2


def test_quota_exhaustion_opens_a_circuit_for_every_later_call() -> None:
    model = ScriptedChatModel([_err(ErrorKind.QUOTA_EXHAUSTED), _ok(), _ok()])
    client, _ = _client(model)

    async def main() -> None:
        for _ in range(3):
            with pytest.raises(ModelCallError) as info:
                await client.complete(REQUEST)
            assert info.value.kind is ErrorKind.QUOTA_EXHAUSTED

    asyncio.run(main())
    assert len(model.requests) == 1  # only the first call reached the provider


def test_a_cached_response_is_replayed_without_calling_the_model(tmp_path: Path) -> None:
    model = ScriptedChatModel([_ok("first"), _ok("second")])
    client, _ = _client(model, cache=ResponseCache(tmp_path))

    async def main() -> tuple[str, bool, str, bool]:
        a = await client.complete(REQUEST)
        b = await client.complete(REQUEST)
        return a.response.text, a.cache_hit, b.response.text, b.cache_hit

    assert asyncio.run(main()) == ("first", False, "first", True)
    assert len(model.requests) == 1
    assert client.stats()["cache_hits"] == 1


def test_identical_requests_in_flight_share_one_call() -> None:
    model = ScriptedChatModel([_ok("shared")], delay_s=0.05)
    client, _ = _client(model)

    async def main() -> list[str]:
        results = await asyncio.gather(*(client.complete(REQUEST) for _ in range(5)))
        return [r.response.text for r in results]

    assert asyncio.run(main()) == ["shared"] * 5
    assert len(model.requests) == 1


def test_rate_limits_and_successes_feed_the_shared_limiter() -> None:
    limiter = AdaptiveRateLimiter(requests_per_minute=6000, tokens_per_minute=None)
    model = ScriptedChatModel([_err(ErrorKind.RATE_LIMITED, 0.0), _ok()])
    client, _ = _client(model, limiter=limiter)
    asyncio.run(client.complete(REQUEST))
    stats = limiter.stats()
    assert stats["throttle_events"] == 1
    assert stats["rate_fraction"] == pytest.approx(0.55)  # halved, then one success step
