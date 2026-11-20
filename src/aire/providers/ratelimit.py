"""Adaptive token bucket shared by every concurrent call to one provider.

Two buckets refill continuously: one in requests (from requests_per_minute),
one in tokens (from tokens_per_minute). Each holds burst_s seconds of budget.
A caller waits until both can cover its request, first come first served.
A request bigger than the token bucket waits for a full bucket and then
leaves it in debt, so the long-run rate is still respected.

The allowed rate adapts: a 429 multiplies it by decrease_factor (never below
min_fraction of the configured rate), drains the buckets and pauses every
caller until the server's retry time; each success adds increase_fraction of
the configured rate back, up to the configured ceiling. This is the
additive-increase, multiplicative-decrease rule.

clock and sleep are injectable so tests run on virtual time.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any


class _Bucket:
    def __init__(self, per_minute: float, burst_s: float, now: float) -> None:
        self.per_second = per_minute / 60.0
        self.capacity = max(1.0, self.per_second * burst_s)
        self.level = self.capacity
        self.updated = now

    def refill(self, now: float, fraction: float) -> None:
        elapsed = max(0.0, now - self.updated)
        self.level = min(self.capacity, self.level + elapsed * self.per_second * fraction)
        self.updated = now

    def wait_for(self, amount: float, fraction: float) -> float:
        """Seconds until the bucket can cover `amount` (capped at its capacity)."""
        need = min(amount, self.capacity)
        if self.level >= need:
            return 0.0
        return (need - self.level) / (self.per_second * fraction)


class AdaptiveRateLimiter:
    def __init__(
        self,
        requests_per_minute: float | None,
        tokens_per_minute: float | None,
        *,
        burst_s: float = 1.0,
        decrease_factor: float = 0.5,
        increase_fraction: float = 0.05,
        min_fraction: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if not 0.0 < decrease_factor < 1.0:
            raise ValueError("decrease_factor must be in (0, 1)")
        if not 0.0 < min_fraction <= 1.0:
            raise ValueError("min_fraction must be in (0, 1]")
        self._clock = clock
        self._sleep = sleep
        now = clock()
        self._requests = _Bucket(requests_per_minute, burst_s, now) if requests_per_minute else None
        self._tokens = _Bucket(tokens_per_minute, burst_s, now) if tokens_per_minute else None
        self._tokens_per_minute = tokens_per_minute
        self._decrease = decrease_factor
        self._increase = increase_fraction
        self._floor = min_fraction
        self._fraction = 1.0
        self._resume_at = now
        self._lock = asyncio.Lock()
        self._throttle_events = 0
        self._waited_s = 0.0

    def _refill(self) -> float:
        now = self._clock()
        for bucket in (self._requests, self._tokens):
            if bucket is not None:
                bucket.refill(now, self._fraction)
        return now

    async def acquire(self, tokens: int) -> float:
        """Wait for budget for one request costing `tokens`; return seconds waited."""
        if self._tokens_per_minute is not None and tokens > self._tokens_per_minute:
            raise ValueError(
                f"request needs {tokens} tokens, more than the {self._tokens_per_minute:g} "
                "tokens per minute budget; it can never be sent"
            )
        waited = 0.0
        async with self._lock:
            while True:
                now = self._refill()
                wait = max(0.0, self._resume_at - now)
                if self._requests is not None:
                    wait = max(wait, self._requests.wait_for(1.0, self._fraction))
                if self._tokens is not None:
                    wait = max(wait, self._tokens.wait_for(float(tokens), self._fraction))
                if wait <= 1e-9:
                    break
                await self._sleep(wait)
                waited += wait
            if self._requests is not None:
                self._requests.level -= 1.0
            if self._tokens is not None:
                self._tokens.level -= float(tokens)
        self._waited_s += waited
        return waited

    def reconcile(self, estimated: int, actual: int) -> None:
        """Correct the token bucket once the real usage of a request is known."""
        if self._tokens is None:
            return
        self._refill()
        self._tokens.level = min(self._tokens.capacity, self._tokens.level + (estimated - actual))

    def on_rate_limited(self, retry_after_s: float | None) -> None:
        """The provider answered 429: slow down and pause until it says to resume."""
        now = self._refill()
        self._fraction = max(self._floor, self._fraction * self._decrease)
        for bucket in (self._requests, self._tokens):
            if bucket is not None:
                bucket.level = min(bucket.level, 0.0)
        if retry_after_s is not None:
            self._resume_at = max(self._resume_at, now + retry_after_s)
        self._throttle_events += 1

    def on_success(self) -> None:
        """A call went through: grow the rate back towards the configured ceiling."""
        self._refill()
        self._fraction = min(1.0, self._fraction + self._increase)

    def stats(self) -> dict[str, float]:
        return {
            "rate_fraction": round(self._fraction, 6),
            "throttle_events": self._throttle_events,
            "waited_s": round(self._waited_s, 6),
        }
