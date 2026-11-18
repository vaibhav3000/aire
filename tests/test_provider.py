"""OpenAI-compatible provider: which failures are retried, and how long it waits.

Every test talks to a scripted HTTP server on 127.0.0.1 with a dummy key, and
the provider's sleep is replaced by a recorder, so nothing waits for real and
nothing leaves the machine.
"""

import json
import random
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from aire.llm.provider import OpenAICompatProvider, ProviderError

KEY_ENV = "AIRE_TEST_PROVIDER_KEY"


def _ok(content: str = "Refunds take 30 days [doc:doc_a].", usage: Any = "default") -> dict:
    body: dict[str, Any] = {
        "model": "fake-model",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
    }
    body["usage"] = {"prompt_tokens": 120, "completion_tokens": 9} if usage == "default" else usage
    return {"status": 200, "body": body}


class _FakeServer:
    """Answers POSTs from a script of responses: {status, body, headers, delay}."""

    def __init__(self) -> None:
        self.script: list[dict[str, Any]] = []
        self.requests = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                server.requests += 1
                step = server.script.pop(0) if server.script else {"status": 500, "body": {}}
                time.sleep(step.get("delay", 0.0))
                payload = json.dumps(step.get("body", {})).encode("utf-8")
                try:
                    self.send_response(step["status"])
                    for name, value in step.get("headers", {}).items():
                        self.send_header(name, value)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # the client gave up (timeout test); nothing to answer

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeServer]:
    monkeypatch.setenv(KEY_ENV, "dummy-test-key")
    fake = _FakeServer()
    yield fake
    fake.close()


def _provider(server: _FakeServer, sleeps: list[float], **kwargs: Any) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        base_url=server.url,
        api_key_env=KEY_ENV,
        model="fake-model",
        timeout_s=kwargs.pop("timeout_s", 5.0),
        sleep=sleeps.append,
        rng=random.Random(0),
        **kwargs,
    )


def test_a_successful_call_reports_tokens_and_one_attempt(server: _FakeServer) -> None:
    server.script = [_ok()]
    response = _provider(server, []).chat("system", "user")
    assert response.content == "Refunds take 30 days [doc:doc_a]."
    assert (response.prompt_tokens, response.completion_tokens) == (120, 9)
    assert response.token_source == "api"
    assert (response.attempts, response.backoff_s) == (1, 0.0)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(server: _FakeServer, status: int) -> None:
    server.script = [{"status": status, "body": {"error": {"message": "bad request"}}}]
    sleeps: list[float] = []
    with pytest.raises(ProviderError) as info:
        _provider(server, sleeps).chat("system", "user")
    assert server.requests == 1
    assert sleeps == []
    assert info.value.status == status


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retried(server: _FakeServer, status: int) -> None:
    server.script = [{"status": status, "body": {}}, _ok()]
    sleeps: list[float] = []
    response = _provider(server, sleeps).chat("system", "user")
    assert server.requests == 2
    assert len(sleeps) == 1
    assert response.attempts == 2


def test_retry_after_header_sets_the_wait(server: _FakeServer) -> None:
    server.script = [{"status": 429, "body": {}, "headers": {"Retry-After": "2"}}, _ok()]
    sleeps: list[float] = []
    response = _provider(server, sleeps).chat("system", "user")
    assert sleeps == [2.0]
    assert response.backoff_s == 2.0


def test_gemini_retry_delay_sets_the_wait(server: _FakeServer) -> None:
    body = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3s"}],
        }
    }
    server.script = [{"status": 429, "body": body}, _ok()]
    sleeps: list[float] = []
    _provider(server, sleeps).chat("system", "user")
    assert sleeps == [3.0]


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}},
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
                        ],
                    }
                ],
            }
        },
    ],
)
def test_exhausted_quota_fails_fast(server: _FakeServer, body: dict) -> None:
    server.script = [{"status": 429, "body": body}, _ok()]
    sleeps: list[float] = []
    with pytest.raises(ProviderError) as info:
        _provider(server, sleeps).chat("system", "user")
    assert server.requests == 1
    assert sleeps == []
    assert info.value.status == 429


def test_requested_wait_beyond_the_cap_fails_fast(server: _FakeServer) -> None:
    server.script = [{"status": 429, "body": {}, "headers": {"Retry-After": "120"}}, _ok()]
    sleeps: list[float] = []
    with pytest.raises(ProviderError):
        _provider(server, sleeps, max_backoff_s=30.0).chat("system", "user")
    assert server.requests == 1
    assert sleeps == []


def test_backoff_grows_exponentially_with_jitter(server: _FakeServer) -> None:
    server.script = [{"status": 503, "body": {}} for _ in range(4)]
    sleeps: list[float] = []
    provider = _provider(server, sleeps, max_retries=3, backoff_base_s=1.0, max_backoff_s=30.0)
    with pytest.raises(ProviderError) as info:
        provider.chat("system", "user")
    assert server.requests == 4
    assert len(sleeps) == 3
    for attempt, wait in enumerate(sleeps):
        ceiling = 2.0**attempt
        assert ceiling / 2 <= wait <= ceiling
    assert info.value.attempts == 4
    assert info.value.backoff_s == pytest.approx(sum(sleeps))


def test_a_timeout_is_retried(server: _FakeServer) -> None:
    server.script = [{"status": 200, "body": _ok()["body"], "delay": 1.0}, _ok()]
    sleeps: list[float] = []
    response = _provider(server, sleeps, timeout_s=0.3).chat("system", "user")
    assert server.requests == 2
    assert len(sleeps) == 1
    assert response.attempts == 2


def test_a_response_without_choices_is_a_provider_error(server: _FakeServer) -> None:
    server.script = [{"status": 200, "body": {"model": "fake-model", "choices": []}}]
    with pytest.raises(ProviderError):
        _provider(server, []).chat("system", "user")
    assert server.requests == 1


def test_a_response_without_usage_reports_unknown_tokens(server: _FakeServer) -> None:
    server.script = [_ok(usage=None)]
    response = _provider(server, []).chat("system", "user")
    assert (response.prompt_tokens, response.completion_tokens) == (0, 0)
    assert response.token_source == "none"


def test_worst_case_duration_bounds_every_wait() -> None:
    provider = OpenAICompatProvider(
        base_url="http://127.0.0.1:9",
        api_key_env=KEY_ENV,
        model="m",
        timeout_s=60.0,
        max_retries=3,
        max_backoff_s=30.0,
    )
    assert provider.max_call_duration_s == 4 * 60.0 + 3 * 30.0
