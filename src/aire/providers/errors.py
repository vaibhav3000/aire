"""Failure classification shared by every adapter.

An adapter makes one attempt and turns any failure into a ProviderError
with a kind. The model client decides what to do from the kind alone:

- rate_limited, transient, timeout, connection: retried with backoff
- quota_exhausted: fail now and stop calling the provider for this run
- client, malformed: fail now; repeating the same request cannot help
"""

from __future__ import annotations

import email.utils
import re
import time
from collections.abc import Mapping
from email.message import Message
from enum import StrEnum

_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
_DAILY_QUOTA_RE = re.compile(r'"quotaId"\s*:\s*"[^"]*PerDay', re.IGNORECASE)
_QUOTA_MARKERS = ("insufficient_quota", "billing_error", "credit balance is too low")


class ErrorKind(StrEnum):
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    CLIENT = "client"
    MALFORMED = "malformed"


RETRYABLE_KINDS = frozenset(
    {ErrorKind.RATE_LIMITED, ErrorKind.TRANSIENT, ErrorKind.TIMEOUT, ErrorKind.CONNECTION}
)


class ProviderError(RuntimeError):
    """One failed attempt, classified."""

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind,
        status: int | None = None,
        retry_after_s: float | None = None,
        provider: str = "",
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.retry_after_s = retry_after_s
        self.provider = provider

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS


def _header(headers: Mapping[str, str] | Message | None, name: str) -> str | None:
    if headers is None:
        return None
    if isinstance(headers, Message):
        return headers.get(name)
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def retry_after_seconds(headers: Mapping[str, str] | Message | None, body: str) -> float | None:
    """The wait a server asked for, in seconds, or None when it gave none.

    Reads, in order: OpenAI's retry-after-ms, Retry-After as seconds or as an
    HTTP date, and Gemini's retryDelay in the error body.
    """
    millis = _header(headers, "retry-after-ms")
    if millis is not None:
        try:
            return max(0.0, float(millis) / 1000.0)
        except ValueError:
            pass  # malformed hint: fall through to the other sources
    value = _header(headers, "retry-after")
    if value is not None:
        try:
            return max(0.0, float(value))
        except ValueError:
            parsed = email.utils.parsedate_to_datetime(value) if value.strip() else None
            if parsed is not None:
                return max(0.0, parsed.timestamp() - time.time())
    match = _RETRY_DELAY_RE.search(body)
    return float(match.group(1)) if match else None


def quota_exhausted(body: str) -> bool:
    """True when a 429 (or billing error) says the allowance is spent, not throttled."""
    return any(marker in body for marker in _QUOTA_MARKERS) or (
        _DAILY_QUOTA_RE.search(body) is not None
    )


def classify_http_status(
    status: int,
    body: str,
    headers: Mapping[str, str] | Message | None,
    *,
    provider: str = "",
) -> ProviderError:
    """Turn an HTTP error status and body into a classified ProviderError."""
    excerpt = body[:300]
    message = f"HTTP {status}: {excerpt}"
    wait = retry_after_seconds(headers, body)
    if status == 429:
        kind = ErrorKind.QUOTA_EXHAUSTED if quota_exhausted(body) else ErrorKind.RATE_LIMITED
    elif status == 408:
        kind = ErrorKind.TIMEOUT
    elif status >= 500:
        kind = ErrorKind.TRANSIENT
    elif status == 402 or (status == 400 and quota_exhausted(body)):
        kind = ErrorKind.QUOTA_EXHAUSTED
    else:
        kind = ErrorKind.CLIENT
    return ProviderError(message, kind=kind, status=status, retry_after_s=wait, provider=provider)
