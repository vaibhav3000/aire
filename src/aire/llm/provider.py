"""Minimal OpenAI-compatible chat client for AIRE (stdlib only).

One provider class for any OpenAI-style endpoint; the API key is read from the
environment at call time and never persisted.

Retry policy:
- Retried: HTTP 408, 429, 500, 502, 503 and 504, timeouts and connection
  failures. Every other status (400, 401, 403, 404, 422, ...) fails at once.
- A 429 that reports an exhausted quota (OpenAI "insufficient_quota", or a
  Gemini per-day quota) fails at once: waiting will not help today.
- The wait honours the server's Retry-After header or Gemini's retryDelay;
  otherwise it is exponential backoff with jitter. A wait longer than
  max_backoff_s fails the call instead of sleeping past the budget.
- Malformed success responses (no choices, invalid JSON) are not retried.
Every response and error carries the number of attempts and the total time
spent waiting between them, so callers can keep that time out of latency.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from email.message import Message

RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
_DAILY_QUOTA_RE = re.compile(r'"quotaId"\s*:\s*"[^"]*PerDay', re.IGNORECASE)


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    finish_reason: str
    token_source: str = "api"  # "none" when the response carried no usage block
    attempts: int = 1
    backoff_s: float = 0.0


class ProviderError(RuntimeError):
    """A failed chat call, with what the retry loop spent on it."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        attempts: int = 1,
        backoff_s: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.attempts = attempts
        self.backoff_s = backoff_s


class _Retry(Exception):
    """Internal: a retryable failure with an optional server-requested wait."""

    def __init__(self, error: ProviderError, wait_s: float | None = None) -> None:
        super().__init__(str(error))
        self.error = error
        self.wait_s = wait_s


def _retry_after_seconds(headers: Message | None, body: str) -> float | None:
    """Wait requested by the server: Retry-After (seconds) or Gemini's retryDelay."""
    if headers is not None:
        value = headers.get("Retry-After")
        if value is not None:
            try:
                return max(0.0, float(value))
            except ValueError:
                pass  # an HTTP-date; fall through to the body and our own backoff
    match = _RETRY_DELAY_RE.search(body)
    return float(match.group(1)) if match else None


def _quota_exhausted(body: str) -> bool:
    return "insufficient_quota" in body or _DAILY_QUOTA_RE.search(body) is not None


class OpenAICompatProvider:
    """Chat-completions client for OpenAI-compatible endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        model: str,
        temperature: float = 0.0,
        timeout_s: float = 90.0,
        max_tokens: int = 1024,
        max_retries: int = 3,
        backoff_base_s: float = 2.0,
        max_backoff_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.max_backoff_s = max_backoff_s
        self._sleep = sleep
        self._rng = rng or random.Random()

    @property
    def max_call_duration_s(self) -> float:
        """Upper bound on one chat() call: every attempt times out and every wait is capped."""
        return (self.max_retries + 1) * self.timeout_s + self.max_retries * self.max_backoff_s

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ProviderError(
                f"environment variable {self.api_key_env} is not set; no API key "
                "available. Deterministic mode needs no key."
            )
        return key

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with equal jitter: a wait in [ceiling/2, ceiling]."""
        ceiling = min(self.max_backoff_s, self.backoff_base_s * (2**attempt))
        return ceiling / 2 + self._rng.uniform(0.0, ceiling / 2)

    def _attempt(self, payload: bytes) -> LLMResponse:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key()}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            error = ProviderError(f"HTTP {exc.code}: {body[:300]}", status=exc.code)
            if exc.code not in RETRYABLE_STATUSES or (exc.code == 429 and _quota_exhausted(body)):
                raise error from exc
            raise _Retry(error, _retry_after_seconds(exc.headers, body)) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise _Retry(ProviderError(f"{type(exc).__name__}: {exc}")) from exc
        return self._parse(raw)

    def _parse(self, raw: bytes) -> LLMResponse:
        try:
            data = json.loads(raw)
            choice = data["choices"][0]
            message = choice.get("message") or {}
            usage = data.get("usage")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError(f"malformed response: {type(exc).__name__}: {exc}") from exc
        content = message.get("content")
        if not content and choice.get("finish_reason") == "length":
            # Thinking models can exhaust max_tokens before emitting content;
            # retrying without progress would waste quota.
            raise ProviderError("response truncated before content (finish_reason=length)")
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            token_source = "api"
        else:
            prompt_tokens = completion_tokens = 0
            token_source = "none"
        return LLMResponse(
            content=content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason") or "",
            token_source=token_source,
        )

    def chat(self, system: str, user: str, retries: int | None = None) -> LLMResponse:
        """One chat completion with the retry policy in the module docstring."""
        max_retries = self.max_retries if retries is None else retries
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        waited = 0.0
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._attempt(payload)
            except ProviderError as error:
                error.attempts, error.backoff_s = attempt, waited
                raise
            except _Retry as retry:
                error = retry.error
                error.attempts, error.backoff_s = attempt, waited
                if attempt > max_retries:
                    raise error from retry
                wait = retry.wait_s if retry.wait_s is not None else self._backoff(attempt - 1)
                if wait > self.max_backoff_s:
                    raise ProviderError(
                        f"{error} (server asked to wait {wait:g}s, above the "
                        f"{self.max_backoff_s:g}s cap)",
                        status=error.status,
                        attempts=attempt,
                        backoff_s=waited,
                    ) from retry
                self._sleep(wait)
                waited += wait
                continue
            response.attempts, response.backoff_s = attempt, waited
            return response
