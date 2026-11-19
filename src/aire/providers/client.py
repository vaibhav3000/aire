"""ModelClient: the one place that decides how a model is called.

Shared by every concurrent case of a run, it puts in front of a one-attempt
ChatModel adapter:

1. the response cache (an exact-request hit never reaches the provider),
2. single flight (identical requests in flight share one call),
3. the adaptive rate limiter (shared RPM and TPM budget),
4. retries for rate limits, transient errors, timeouts and connection
   failures: the server's requested wait when it gives one, otherwise
   exponential backoff with full jitter, uniform(0, min(cap, base * 2**n)),
5. a quota circuit: once the provider reports an exhausted quota, every
   later call fails immediately without touching the network.

Every attempt is returned as an AttemptRecord, so the system under test can
record one span per attempt and keep waits out of its latency (ADR 0001).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .base import ChatModel
from .cache import ResponseCache, request_key
from .errors import ErrorKind, ProviderError
from .ratelimit import AdaptiveRateLimiter
from .types import ChatRequest, ChatResponse


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0  # longest single wait; a server asking for more fails the call


@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    started_at: float  # unix seconds
    latency_ms: float
    backoff_s: float = 0.0  # retry wait before this attempt
    queued_s: float = 0.0  # rate-limiter wait before this attempt
    error_kind: str | None = None
    status: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class Completion:
    response: ChatResponse
    attempts: list[AttemptRecord] = field(default_factory=list)
    cache_hit: bool = False
    waited_s: float = 0.0  # backoff plus rate-limiter waits, never part of latency
    cached_latency_ms: float | None = None  # latency of the original call on a cache hit


class ModelCallError(ProviderError):
    """The request failed for good; carries every attempt that was made."""

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind,
        status: int | None = None,
        provider: str = "",
        attempts: list[AttemptRecord] | None = None,
        waited_s: float = 0.0,
    ) -> None:
        super().__init__(message, kind=kind, status=status, provider=provider)
        self.attempts = list(attempts or [])
        self.waited_s = waited_s


class ModelClient:
    def __init__(
        self,
        model: ChatModel,
        *,
        limiter: AdaptiveRateLimiter | None = None,
        cache: ResponseCache | None = None,
        retry: RetryPolicy | None = None,
        rng: random.Random | None = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.model = model
        self.limiter = limiter
        self.cache = cache
        self.retry = retry or RetryPolicy()
        self._rng = rng or random.Random()
        self._sleep = sleep
        self._inflight: dict[str, asyncio.Future[Completion]] = {}
        self._quota_error: str | None = None
        self._stats = {
            "calls": 0,
            "cache_hits": 0,
            "shared": 0,
            "attempts": 0,
            "rate_limited": 0,
            "failures": 0,
        }

    @property
    def provider(self) -> str:
        return self.model.provider

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    async def aclose(self) -> None:
        await self.model.aclose()

    async def complete(self, request: ChatRequest) -> Completion:
        self._stats["calls"] += 1
        if self._quota_error is not None:
            self._stats["failures"] += 1
            raise ModelCallError(
                f"not sent: the provider reported an exhausted quota earlier in this run "
                f"({self._quota_error})",
                kind=ErrorKind.QUOTA_EXHAUSTED,
                provider=self.provider,
            )
        key = request_key(self.provider, request)
        if self.cache is not None:
            entry = self.cache.get(key)
            if entry is not None:
                self._stats["cache_hits"] += 1
                return Completion(
                    response=entry.response, cache_hit=True, cached_latency_ms=entry.latency_ms
                )
        shared = self._inflight.get(key)
        if shared is not None:
            self._stats["shared"] += 1
            return await asyncio.shield(shared)

        future: asyncio.Future[Completion] = asyncio.get_running_loop().create_future()
        # Mark the outcome as observed even when nobody else waits on it.
        future.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        self._inflight[key] = future
        try:
            completion = await self._call_with_retries(request, key)
        except ProviderError as error:
            future.set_exception(error)
            raise
        except asyncio.CancelledError:
            future.set_exception(
                ProviderError("the shared call was cancelled", kind=ErrorKind.TRANSIENT)
            )
            raise
        else:
            future.set_result(completion)
            return completion
        finally:
            del self._inflight[key]

    def _backoff(self, attempt: int) -> float:
        ceiling = min(self.retry.max_delay_s, self.retry.base_delay_s * 2 ** (attempt - 1))
        return self._rng.uniform(0.0, ceiling)

    async def _call_with_retries(self, request: ChatRequest, key: str) -> Completion:
        attempts: list[AttemptRecord] = []
        waited = 0.0
        backoff = 0.0
        estimate = request.estimated_tokens()
        for attempt in range(1, self.retry.max_attempts + 1):
            queued = 0.0
            if self.limiter is not None:
                try:
                    queued = await self.limiter.acquire(estimate)
                except ValueError as exc:
                    self._stats["failures"] += 1
                    raise ModelCallError(
                        str(exc), kind=ErrorKind.CLIENT, provider=self.provider
                    ) from exc
                waited += queued
            self._stats["attempts"] += 1
            started_at = time.time()
            start = time.perf_counter()
            try:
                response = await self.model.complete(request)
            except ProviderError as error:
                latency_ms = (time.perf_counter() - start) * 1000.0
                attempts.append(
                    AttemptRecord(
                        attempt=attempt,
                        started_at=started_at,
                        latency_ms=latency_ms,
                        backoff_s=backoff,
                        queued_s=queued,
                        error_kind=str(error.kind),
                        status=error.status,
                        error=str(error)[:500],
                    )
                )
                delay = self._next_delay(error, attempt, attempts, waited)
                await self._sleep(delay)
                waited += delay
                backoff = delay
                continue
            latency_ms = (time.perf_counter() - start) * 1000.0
            attempts.append(
                AttemptRecord(
                    attempt=attempt,
                    started_at=started_at,
                    latency_ms=latency_ms,
                    backoff_s=backoff,
                    queued_s=queued,
                )
            )
            if self.limiter is not None:
                actual = response.usage.input_tokens + response.usage.output_tokens
                self.limiter.reconcile(
                    estimate, actual if response.usage.source == "api" else estimate
                )
                self.limiter.on_success()
            if self.cache is not None:
                self.cache.put(key, response, latency_ms)
            return Completion(response=response, attempts=attempts, waited_s=waited)
        raise AssertionError("unreachable: the retry loop always returns or raises")

    def _next_delay(
        self,
        error: ProviderError,
        attempt: int,
        attempts: list[AttemptRecord],
        waited: float,
    ) -> float:
        """The wait before the next attempt, or raise ModelCallError when there is none."""

        def fail(message: str) -> ModelCallError:
            self._stats["failures"] += 1
            return ModelCallError(
                message,
                kind=error.kind,
                status=error.status,
                provider=self.provider,
                attempts=attempts,
                waited_s=waited,
            )

        if error.kind is ErrorKind.RATE_LIMITED:
            self._stats["rate_limited"] += 1
            if self.limiter is not None:
                self.limiter.on_rate_limited(error.retry_after_s)
        if error.kind is ErrorKind.QUOTA_EXHAUSTED:
            self._quota_error = str(error)[:200]
            raise fail(f"quota exhausted: {error}")
        if not error.retryable:
            raise fail(str(error))
        if attempt >= self.retry.max_attempts:
            raise fail(f"{error} (gave up after {attempt} attempts)")
        delay = error.retry_after_s if error.retry_after_s is not None else self._backoff(attempt)
        if delay > self.retry.max_delay_s:
            raise fail(
                f"{error} (server asked to wait {delay:g}s, above the "
                f"{self.retry.max_delay_s:g}s cap)"
            )
        return delay
