"""Adaptive token bucket: pacing, 429 back-off, recovery and token reconciliation.

A virtual clock replaces time, so every wait is exact and nothing sleeps.
"""

import asyncio

import pytest

from aire.providers.ratelimit import AdaptiveRateLimiter


class VirtualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)


def _limiter(clock: VirtualClock, **kwargs: object) -> AdaptiveRateLimiter:
    return AdaptiveRateLimiter(clock=clock, sleep=clock.sleep, **kwargs)  # type: ignore[arg-type]


def test_requests_are_paced_at_the_configured_rate() -> None:
    clock = VirtualClock()
    limiter = _limiter(clock, requests_per_minute=120, tokens_per_minute=None, burst_s=1.0)

    async def main() -> list[float]:
        return [await limiter.acquire(1) for _ in range(4)]

    waits = asyncio.run(main())
    # 120 RPM = 2 per second with a one-second burst: two at once, then one per 0.5 s.
    assert waits == pytest.approx([0.0, 0.0, 0.5, 0.5])
    assert clock.now == pytest.approx(1.0)


def test_concurrent_callers_share_one_budget() -> None:
    clock = VirtualClock()
    limiter = _limiter(clock, requests_per_minute=60, tokens_per_minute=None, burst_s=1.0)

    async def main() -> None:
        await asyncio.gather(*(limiter.acquire(1) for _ in range(5)))

    asyncio.run(main())
    # One request per second, one allowed at once: the fifth starts at t=4.
    assert clock.now == pytest.approx(4.0)


def test_token_budget_paces_large_requests() -> None:
    clock = VirtualClock()
    limiter = _limiter(clock, requests_per_minute=None, tokens_per_minute=6000, burst_s=1.0)

    async def main() -> list[float]:
        return [await limiter.acquire(100) for _ in range(3)]

    # 100 tokens per second: each 100 token request needs one second of budget.
    assert asyncio.run(main()) == pytest.approx([0.0, 1.0, 1.0])


def test_a_request_larger_than_a_minute_of_tokens_is_refused() -> None:
    clock = VirtualClock()
    limiter = _limiter(clock, requests_per_minute=None, tokens_per_minute=1000)
    with pytest.raises(ValueError, match="tokens per minute"):
        asyncio.run(limiter.acquire(1001))


def test_a_rate_limit_halves_the_rate_and_pauses_everyone() -> None:
    clock = VirtualClock()
    limiter = _limiter(clock, requests_per_minute=120, tokens_per_minute=None, burst_s=1.0)

    async def main() -> float:
        await limiter.acquire(1)
        limiter.on_rate_limited(retry_after_s=3.0)
        return await limiter.acquire(1)

    waited = asyncio.run(main())
    assert limiter.stats()["rate_fraction"] == pytest.approx(0.5)
    assert limiter.stats()["throttle_events"] == 1
    # Nobody starts before the server's 3 s, and the drained bucket refills at
    # the halved rate (1 per second) after that.
    assert waited >= 3.0
    assert clock.now == pytest.approx(waited)


def test_successes_recover_the_rate_but_never_above_the_ceiling() -> None:
    clock = VirtualClock()
    limiter = _limiter(
        clock, requests_per_minute=60, tokens_per_minute=None, increase_fraction=0.25
    )
    limiter.on_rate_limited(retry_after_s=None)
    limiter.on_rate_limited(retry_after_s=None)
    assert limiter.stats()["rate_fraction"] == pytest.approx(0.25)
    for _ in range(10):
        limiter.on_success()
    assert limiter.stats()["rate_fraction"] == pytest.approx(1.0)


def test_the_rate_never_drops_below_the_floor() -> None:
    clock = VirtualClock()
    limiter = _limiter(clock, requests_per_minute=60, tokens_per_minute=None, min_fraction=0.1)
    for _ in range(20):
        limiter.on_rate_limited(retry_after_s=None)
    assert limiter.stats()["rate_fraction"] == pytest.approx(0.1)


def test_reconcile_returns_unused_tokens_to_the_budget() -> None:
    clock = VirtualClock()
    limiter = _limiter(clock, requests_per_minute=None, tokens_per_minute=6000, burst_s=1.0)

    async def main() -> float:
        await limiter.acquire(100)
        limiter.reconcile(estimated=100, actual=20)  # 80 tokens come back
        return await limiter.acquire(80)

    assert asyncio.run(main()) == pytest.approx(0.0)


def test_no_limits_means_no_waiting() -> None:
    clock = VirtualClock()
    limiter = _limiter(clock, requests_per_minute=None, tokens_per_minute=None)

    async def main() -> list[float]:
        return [await limiter.acquire(10_000) for _ in range(50)]

    assert set(asyncio.run(main())) == {0.0}
