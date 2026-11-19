"""Classify exceptions raised by provider SDKs without importing the SDKs.

OpenAI and Anthropic raise APIStatusError subclasses carrying status_code,
the httpx2 response (for Retry-After headers) and the parsed body;
google-genai raises APIError subclasses carrying code (the HTTP status) and
details (the error JSON). Transport failures surface as the SDKs' own
APITimeoutError and APIConnectionError, or as raw httpx2 exceptions from
google-genai. The checks below read only those attributes and class names,
so the core never imports an SDK; the adapter tests exercise them with the
real SDK exception classes.
"""

from __future__ import annotations

import json
from typing import Any

from .errors import ErrorKind, ProviderError, classify_http_status

_TIMEOUT_CLASSES = frozenset({"APITimeoutError", "TimeoutException", "TimeoutError"})
_CONNECTION_CLASSES = frozenset(
    {"APIConnectionError", "TransportError", "NetworkError", "ConnectError", "ConnectionError"}
)


def _class_names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def _status(exc: BaseException) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def _headers(exc: BaseException) -> dict[str, str]:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None or not hasattr(headers, "items"):
        return {}
    return {str(k): str(v) for k, v in headers.items()}


def _body_text(exc: BaseException) -> str:
    body: Any = getattr(exc, "body", None)
    if body is None:
        body = getattr(exc, "details", None)
    if body is None:
        return str(exc)
    text = body if isinstance(body, str) else json.dumps(body, default=str)
    return f"{text} {exc}"


def classify_sdk_exception(exc: BaseException, *, provider: str) -> ProviderError | None:
    """A ProviderError for an SDK failure, or None when `exc` is not one (a bug)."""
    status = _status(exc)
    if status is not None:
        return classify_http_status(status, _body_text(exc), _headers(exc), provider=provider)
    names = _class_names(exc)
    if names & _TIMEOUT_CLASSES:
        return ProviderError(
            f"timeout: {type(exc).__name__}: {exc}", kind=ErrorKind.TIMEOUT, provider=provider
        )
    if names & _CONNECTION_CLASSES:
        return ProviderError(
            f"connection failed: {type(exc).__name__}: {exc}",
            kind=ErrorKind.CONNECTION,
            provider=provider,
        )
    return None


def require_key(env_name: str, provider: str) -> str:
    """The API key from the environment; never logged or stored on a result."""
    import os

    key = os.environ.get(env_name, "").strip()
    if not key:
        raise ProviderError(
            f"environment variable {env_name} is not set; {provider} needs an API key",
            kind=ErrorKind.CLIENT,
            provider=provider,
        )
    return key
