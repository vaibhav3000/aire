"""Stdlib asyncio HTTP client and the local OpenAI-compatible adapter.

A scripted asyncio server on 127.0.0.1 plays the provider; nothing leaves the
machine and no key is real.
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from aire.providers.errors import ErrorKind, ProviderError
from aire.providers.http import post_json
from aire.providers.local import LocalOpenAICompatModel
from aire.providers.types import ChatRequest, Message


@dataclass
class Reply:
    status: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    chunked: bool = False
    hang: bool = False  # never answer; wait for the client to go away


@dataclass
class Received:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


class ScriptedServer:
    def __init__(self, replies: list[Reply]) -> None:
        self.replies = replies
        self.received: list[Received] = []
        self.client_left_after: float | None = None
        self._server: asyncio.Server | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/v1"

    async def __aenter__(self) -> "ScriptedServer":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request_line = (await reader.readline()).decode()
        headers: dict[str, str] = {}
        while (line := await reader.readline()) not in (b"\r\n", b""):
            name, _, value = line.decode().partition(":")
            headers[name.strip().lower()] = value.strip()
        body = await reader.readexactly(int(headers.get("content-length", "0")))
        self.received.append(Received(request_line.split()[1], headers, json.loads(body or b"{}")))
        reply = self.replies.pop(0)
        if reply.hang:
            start = time.monotonic()
            await reader.read()  # returns at EOF, i.e. when the client closes
            self.client_left_after = time.monotonic() - start
            writer.close()
            return
        payload = json.dumps(reply.body).encode() if reply.body is not None else b""
        head = [f"HTTP/1.1 {reply.status} X", "Content-Type: application/json"]
        head += [f"{k}: {v}" for k, v in reply.headers.items()]
        if reply.chunked:
            head.append("Transfer-Encoding: chunked")
            writer.write(("\r\n".join(head) + "\r\n\r\n").encode())
            for i in range(0, len(payload), 7):
                piece = payload[i : i + 7]
                writer.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
            writer.write(b"0\r\n\r\n")
        else:
            head.append(f"Content-Length: {len(payload)}")
            writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + payload)
        await writer.drain()
        writer.close()


def _serve(replies: list[Reply], body: Callable[[ScriptedServer], Awaitable[Any]]) -> Any:
    async def main() -> Any:
        async with ScriptedServer(replies) as server:
            return await body(server)

    return asyncio.run(main())


def _completion(content: str, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "cmpl-1",
        "model": "served-model",
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }
    if usage is not None:
        data["usage"] = usage
    return data


REQUEST = ChatRequest(
    model="m", messages=(Message("system", "sys"), Message("user", "hi")), max_output_tokens=50
)
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
}


def test_post_json_decodes_a_chunked_body() -> None:
    async def body(server: ScriptedServer) -> Any:
        return await post_json(f"{server.url}/x", {"a": 1}, timeout_s=5)

    response = _serve([Reply(body={"hello": "world" * 20}, chunked=True)], body)
    assert response.status == 200
    assert json.loads(response.body) == {"hello": "world" * 20}
    assert response.headers["transfer-encoding"] == "chunked"


def test_the_adapter_sends_an_openai_chat_request_and_maps_usage() -> None:
    usage = {
        "prompt_tokens": 40,
        "completion_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 30},
        "completion_tokens_details": {"reasoning_tokens": 5},
    }

    async def body(server: ScriptedServer) -> Any:
        model = LocalOpenAICompatModel(server.url, provider="ollama")
        response = await model.complete(REQUEST)
        return response, server.received[0]

    response, received = _serve([Reply(body=_completion("Hello", usage))], body)
    assert received.path == "/v1/chat/completions"
    assert received.body["model"] == "m"
    assert received.body["max_tokens"] == 50
    assert received.body["temperature"] == 0.0
    assert received.body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert "authorization" not in received.headers
    assert response.text == "Hello"
    assert response.response_model == "served-model"
    assert response.response_id == "cmpl-1"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (40, 12)
    assert response.usage.cache_read_tokens == 30
    assert response.usage.reasoning_tokens == 5
    assert response.usage.source == "api"


def test_structured_output_is_requested_and_validated() -> None:
    request = ChatRequest(model="m", messages=(Message("user", "x"),), json_schema=SCHEMA)

    async def body(server: ScriptedServer) -> Any:
        model = LocalOpenAICompatModel(server.url)
        ok = await model.complete(request)
        with pytest.raises(ProviderError) as info:
            await model.complete(request)
        return ok, info.value, server.received[0].body

    ok, error, sent = _serve(
        [Reply(body=_completion('{"label": "yes"}')), Reply(body=_completion('{"label": 3}'))],
        body,
    )
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["schema"] == SCHEMA
    assert ok.parsed == {"label": "yes"}
    assert error.kind is ErrorKind.MALFORMED


def test_a_response_without_usage_is_labelled_as_such() -> None:
    async def body(server: ScriptedServer) -> Any:
        return await LocalOpenAICompatModel(server.url).complete(REQUEST)

    response = _serve([Reply(body=_completion("Hello"))], body)
    assert response.usage.source == "none"


def test_a_429_carries_the_server_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRE_TEST_LOCAL_KEY", "dummy-key")

    async def body(server: ScriptedServer) -> Any:
        model = LocalOpenAICompatModel(server.url, api_key_env="AIRE_TEST_LOCAL_KEY")
        with pytest.raises(ProviderError) as info:
            await model.complete(REQUEST)
        return info.value, server.received[0].headers

    error, headers = _serve(
        [Reply(status=429, body={"error": "slow"}, headers={"Retry-After": "3"})], body
    )
    assert error.kind is ErrorKind.RATE_LIMITED
    assert error.retry_after_s == 3.0
    assert headers["authorization"] == "Bearer dummy-key"


def test_a_missing_key_fails_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIRE_TEST_LOCAL_KEY", raising=False)
    model = LocalOpenAICompatModel("http://127.0.0.1:9/v1", api_key_env="AIRE_TEST_LOCAL_KEY")
    with pytest.raises(ProviderError, match="AIRE_TEST_LOCAL_KEY") as info:
        asyncio.run(model.complete(REQUEST))
    assert info.value.kind is ErrorKind.CLIENT


def test_a_refused_connection_is_a_connection_error() -> None:
    async def main() -> ProviderError:
        server = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        server.close()
        await server.wait_closed()
        with pytest.raises(ProviderError) as info:
            await LocalOpenAICompatModel(f"http://127.0.0.1:{port}/v1").complete(REQUEST)
        return info.value

    assert asyncio.run(main()).kind is ErrorKind.CONNECTION


def test_a_timeout_closes_the_connection_instead_of_leaking_it() -> None:
    async def body(server: ScriptedServer) -> Any:
        model = LocalOpenAICompatModel(server.url, timeout_s=0.2)
        with pytest.raises(ProviderError) as info:
            await model.complete(REQUEST)
        for _ in range(100):  # let the server notice the closed socket
            if server.client_left_after is not None:
                break
            await asyncio.sleep(0.02)
        return info.value, server.client_left_after

    error, left_after = _serve([Reply(hang=True)], body)
    assert error.kind is ErrorKind.TIMEOUT
    assert left_after is not None and left_after < 2.0


def test_truncated_output_with_no_content_is_malformed() -> None:
    data = _completion("")
    data["choices"][0]["finish_reason"] = "length"

    async def body(server: ScriptedServer) -> Any:
        with pytest.raises(ProviderError) as info:
            await LocalOpenAICompatModel(server.url).complete(REQUEST)
        return info.value

    assert _serve([Reply(body=data)], body).kind is ErrorKind.MALFORMED
