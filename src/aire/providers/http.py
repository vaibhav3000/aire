"""A minimal asyncio HTTP/1.1 client for JSON POST requests (stdlib only).

Enough for OpenAI-compatible endpoints: http and https, one connection per
request (Connection: close), bodies framed by Content-Length, chunked
transfer encoding or end of stream. The whole exchange runs under one
timeout, and because it is a coroutine a timeout or cancellation closes the
socket, so an abandoned request stops consuming the provider's budget.

Failures are raised as ProviderError with kind timeout, connection or
malformed; HTTP error statuses are returned, not raised, so the caller can
classify them with the response body.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .errors import ErrorKind, ProviderError

MAX_BODY_BYTES = 16 * 1024 * 1024
USER_AGENT = "aire-http/1"


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    headers: dict[str, str]  # names lower-cased
    body: bytes

    def text(self) -> str:
        return self.body.decode("utf-8", "replace")


def _safe_header(name: str, value: str) -> str:
    if any(c in name + value for c in "\r\n"):
        raise ProviderError(f"header {name!r} contains a line break", kind=ErrorKind.CLIENT)
    return f"{name}: {value}\r\n"


async def _read_body(reader: asyncio.StreamReader, headers: Mapping[str, str]) -> bytes:
    if "chunked" in headers.get("transfer-encoding", "").lower():
        parts: list[bytes] = []
        total = 0
        while True:
            size_line = await reader.readline()
            size = int(size_line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                await reader.readline()  # the empty line after the last chunk
                return b"".join(parts)
            total += size
            if total > MAX_BODY_BYTES:
                raise ProviderError("response body too large", kind=ErrorKind.MALFORMED)
            parts.append(await reader.readexactly(size))
            await reader.readline()  # CRLF after each chunk
    if "content-length" in headers:
        length = int(headers["content-length"])
        if length > MAX_BODY_BYTES:
            raise ProviderError("response body too large", kind=ErrorKind.MALFORMED)
        return await reader.readexactly(length)
    body = await reader.read(MAX_BODY_BYTES + 1)
    if len(body) > MAX_BODY_BYTES:
        raise ProviderError("response body too large", kind=ErrorKind.MALFORMED)
    return body


async def _exchange(
    url: str, payload: bytes, headers: Mapping[str, str], ssl_context: ssl.SSLContext | None
) -> HTTPResponse:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ProviderError(f"unsupported URL {url!r}", kind=ErrorKind.CLIENT)
    secure = parts.scheme == "https"
    port = parts.port or (443 if secure else 80)
    context = (ssl_context or ssl.create_default_context()) if secure else None
    reader, writer = await asyncio.open_connection(
        parts.hostname, port, ssl=context, server_hostname=parts.hostname if secure else None
    )
    try:
        path = parts.path or "/"
        if parts.query:
            path += f"?{parts.query}"
        host = parts.hostname if parts.port is None else f"{parts.hostname}:{parts.port}"
        head = f"POST {path} HTTP/1.1\r\n"
        head += _safe_header("Host", host)
        head += _safe_header("User-Agent", USER_AGENT)
        head += _safe_header("Accept", "application/json")
        head += _safe_header("Content-Type", "application/json")
        head += _safe_header("Content-Length", str(len(payload)))
        head += _safe_header("Connection", "close")
        for name, value in headers.items():
            head += _safe_header(name, value)
        writer.write(head.encode("latin-1") + b"\r\n" + payload)
        await writer.drain()

        status_line = await reader.readline()
        fields = status_line.decode("latin-1").split(maxsplit=2)
        if len(fields) < 2 or not fields[0].startswith("HTTP/"):
            raise ProviderError(
                f"malformed status line {status_line[:80]!r}", kind=ErrorKind.MALFORMED
            )
        status = int(fields[1])
        response_headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode("latin-1").partition(":")
            response_headers[name.strip().lower()] = value.strip()
        body = await _read_body(reader, response_headers)
        return HTTPResponse(status=status, headers=response_headers, body=body)
    finally:
        writer.close()
        # The peer may already have dropped the connection; closing is all we need.
        with contextlib.suppress(OSError):
            await writer.wait_closed()


async def post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None = None,
    timeout_s: float,
    ssl_context: ssl.SSLContext | None = None,
) -> HTTPResponse:
    """POST `payload` as JSON and return the response (any status)."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        async with asyncio.timeout(timeout_s):
            return await _exchange(url, data, headers or {}, ssl_context)
    except TimeoutError as exc:
        raise ProviderError(f"no response within {timeout_s:g}s", kind=ErrorKind.TIMEOUT) from exc
    except (asyncio.IncompleteReadError, ValueError) as exc:
        raise ProviderError(
            f"malformed HTTP response: {type(exc).__name__}: {exc}", kind=ErrorKind.MALFORMED
        ) from exc
    except OSError as exc:  # refused, reset, DNS failure, TLS errors
        raise ProviderError(
            f"connection failed: {type(exc).__name__}: {exc}", kind=ErrorKind.CONNECTION
        ) from exc
