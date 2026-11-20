"""FakeLLMServer: a 127.0.0.1 OpenAI-compatible server that behaves like a provider.

It answers POST .../chat/completions with an extractive answer computed from
the RAG prompt (the first document in the context is treated as the best,
exactly as the deterministic demo system does), so a served run is
deterministic and scorable. Around that it behaves like a hosted API:

- requests_per_minute / tokens_per_minute: a token bucket; over budget, it
  answers 429 with a wait hint ("header": retry-after-ms and Retry-After,
  "gemini": a retryDelay in the error body, "none": no hint),
- service_time_s: simulated model latency per request,
- fail_every: every Nth accepted request gets a 503,
- quota_after: after that many answers, every request gets a 429 saying the
  quota is exhausted.

Counters (requests, served, rate_limited, injected_failures, quota_rejected)
let tests and benchmarks check what reached the server.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import re
import time
from typing import Any, Literal

from ..llm.deterministic import extractive_answer
from ..systems.prompts import CONTEXT_HEADER, NO_DOCUMENTS, QUESTION_MARKER

RetryHint = Literal["header", "gemini", "none"]
_DOC_HEADER_RE = re.compile(r"^\[doc:([^\]\n]+)\]\n", re.MULTILINE)
_ABSTAIN = "I don't know."


def answer_from_prompt(user_message: str) -> str:
    """The extractive answer to a RAG user message built by rag_user_prompt."""
    context, marker, question = user_message.rpartition(QUESTION_MARKER)
    if not marker:
        return _ABSTAIN
    context = context.removeprefix(CONTEXT_HEADER)
    if context.strip() == NO_DOCUMENTS:
        return _ABSTAIN
    pieces = _DOC_HEADER_RE.split(context)
    # split() gives ["", id1, text1, id2, text2, ...]
    documents = [(pieces[i], pieces[i + 1].strip()) for i in range(1, len(pieces) - 1, 2)]
    if not documents:
        return _ABSTAIN
    doc_id, text = documents[0]
    return extractive_answer(question, doc_id, text) or _ABSTAIN


class FakeLLMServer:
    def __init__(
        self,
        *,
        requests_per_minute: float | None = None,
        tokens_per_minute: float | None = None,
        burst_s: float = 1.0,
        service_time_s: float = 0.0,
        fail_every: int = 0,
        quota_after: int | None = None,
        retry_hint: RetryHint = "header",
    ) -> None:
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.burst_s = burst_s
        self.service_time_s = service_time_s
        self.fail_every = fail_every
        self.quota_after = quota_after
        self.retry_hint = retry_hint
        self.requests = 0
        self.served = 0
        self.rate_limited = 0
        self.injected_failures = 0
        self.quota_rejected = 0
        self._accepted = 0
        self._server: asyncio.Server | None = None
        now = time.monotonic()
        self._req_level = self._capacity(self.rpm)
        self._tok_level = self._capacity(self.tpm)
        self._updated = now

    def _capacity(self, per_minute: float | None) -> float:
        return max(1.0, per_minute / 60.0 * self.burst_s) if per_minute else math.inf

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("the server is not running; use it as an async context manager")
        port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/v1"

    async def __aenter__(self) -> FakeLLMServer:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    # --- budget -------------------------------------------------------------

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        self._updated = now
        if self.rpm:
            self._req_level = min(
                self._capacity(self.rpm), self._req_level + elapsed * self.rpm / 60
            )
        if self.tpm:
            self._tok_level = min(
                self._capacity(self.tpm), self._tok_level + elapsed * self.tpm / 60
            )

    def _admit(self, tokens: int) -> float | None:
        """None when admitted; otherwise the seconds until the budget allows it."""
        self._refill()
        waits = []
        if self.rpm and self._req_level < 1.0:
            waits.append((1.0 - self._req_level) / (self.rpm / 60))
        if self.tpm:
            need = min(float(tokens), self._capacity(self.tpm))
            if self._tok_level < need:
                waits.append((need - self._tok_level) / (self.tpm / 60))
        if waits:
            return max(waits)
        if self.rpm:
            self._req_level -= 1.0
        if self.tpm:
            self._tok_level -= float(tokens)
        return None

    # --- HTTP ---------------------------------------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            headers: dict[str, str] = {}
            while (line := await reader.readline()) not in (b"\r\n", b"\n", b""):
                name, _, value = line.decode("latin-1").partition(":")
                headers[name.strip().lower()] = value.strip()
            body = await reader.readexactly(int(headers.get("content-length", "0")))
            status, extra, payload = await self._respond(request_line.decode("latin-1"), body)
            data = json.dumps(payload).encode("utf-8")
            head = [f"HTTP/1.1 {status} {'OK' if status == 200 else 'Error'}"]
            head += ["Content-Type: application/json", f"Content-Length: {len(data)}"]
            head += [f"{k}: {v}" for k, v in extra.items()]
            head.append("Connection: close")
            writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + data)
            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass  # the client went away (timeout or cancellation); nothing to answer
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()

    async def _respond(self, request_line: str, body: bytes) -> tuple[int, dict[str, str], Any]:
        self.requests += 1
        parts = request_line.split()
        if len(parts) < 2 or parts[0] != "POST" or not parts[1].endswith("/chat/completions"):
            return 404, {}, {"error": {"message": "not found"}}
        try:
            request = json.loads(body)
            messages = request["messages"]
            user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        except (ValueError, KeyError, StopIteration, TypeError) as exc:
            return 400, {}, {"error": {"message": f"bad request: {exc}"}}
        if "response_format" in request:
            return 400, {}, {"error": {"message": "structured output is not supported here"}}

        if self.quota_after is not None and self.served >= self.quota_after:
            self.quota_rejected += 1
            return 429, {}, {"error": {"code": "insufficient_quota", "message": "quota exhausted"}}
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        wait = self._admit(prompt_tokens + int(request.get("max_tokens") or 0))
        if wait is not None:
            self.rate_limited += 1
            return 429, *self._rate_limit_reply(wait)
        self._accepted += 1
        if self.fail_every and self._accepted % self.fail_every == 0:
            self.injected_failures += 1
            return 503, {}, {"error": {"message": "injected failure"}}

        if self.service_time_s:
            await asyncio.sleep(self.service_time_s)
        answer = answer_from_prompt(user)
        self.served += 1
        return (
            200,
            {},
            {
                "id": f"fake-{self.served}",
                "object": "chat.completion",
                "model": request.get("model", "fake"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": max(1, len(answer) // 4),
                    "total_tokens": prompt_tokens + max(1, len(answer) // 4),
                },
            },
        )

    def _rate_limit_reply(self, wait: float) -> tuple[dict[str, str], Any]:
        error: dict[str, Any] = {"code": "rate_limit_exceeded", "message": "slow down"}
        if self.retry_hint == "header":
            millis = math.ceil(wait * 1000)
            return {"retry-after-ms": str(millis), "Retry-After": str(math.ceil(wait))}, {
                "error": error
            }
        if self.retry_hint == "gemini":
            details = [{"@type": "RetryInfo", "retryDelay": f"{math.ceil(wait * 1000) / 1000}s"}]
            return {}, {"error": {**error, "details": details}}
        return {}, {"error": error}
