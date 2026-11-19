"""Gemini, OpenAI and Anthropic adapters, faked at the client seam (ADR 0002).

The SDK clients are replaced by small fakes that record the call and return
real SDK response objects; errors are the SDKs' own exception classes. No
request leaves the machine and no key is real.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from anthropic.types import Message as AnthropicMessage
from google.genai import errors as genai_errors
from openai.types.chat import ChatCompletion

from aire.providers.anthropic_sdk import AnthropicModel
from aire.providers.errors import ErrorKind, ProviderError
from aire.providers.gemini_sdk import GeminiModel
from aire.providers.openai_sdk import OpenAIModel
from aire.providers.types import ChatRequest, Message

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
}
REQUEST = ChatRequest(
    model="m",
    messages=(Message("system", "Be brief."), Message("user", "Classify this.")),
    max_output_tokens=64,
    temperature=0.0,
)
STRUCTURED = ChatRequest(
    model="m", messages=(Message("user", "Classify this."),), json_schema=SCHEMA
)


def _http_response(status: int, headers: dict[str, str] | None = None) -> httpx2.Response:
    return httpx2.Response(
        status, headers=headers or {}, request=httpx2.Request("POST", "https://example.invalid")
    )


class Recorder:
    """An async callable that records its kwargs and returns or raises a scripted item."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


# --- OpenAI -------------------------------------------------------------------


def _openai_client(result: Any) -> tuple[Any, Recorder]:
    create = Recorder(result)

    async def close() -> None:
        return None

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close
    )
    return client, create


def _openai_completion(content: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m-2026",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70,
                "prompt_tokens_details": {"cached_tokens": 32},
                "completion_tokens_details": {"reasoning_tokens": 8},
            },
        }
    )


def test_openai_request_and_usage_mapping() -> None:
    client, create = _openai_client(_openai_completion("positive"))
    response = asyncio.run(OpenAIModel(client=client).complete(REQUEST))
    (call,) = create.calls
    assert call["model"] == "m"
    assert call["max_completion_tokens"] == 64
    assert "max_tokens" not in call
    assert call["temperature"] == 0.0
    assert call["messages"][0] == {"role": "system", "content": "Be brief."}
    assert response.text == "positive"
    assert response.response_model == "m-2026"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (50, 20)
    assert response.usage.cache_read_tokens == 32
    assert response.usage.reasoning_tokens == 8


def test_openai_structured_output_is_strict_and_validated() -> None:
    client, create = _openai_client(_openai_completion('{"label": "spam"}'))
    response = asyncio.run(OpenAIModel(client=client).complete(STRUCTURED))
    fmt = create.calls[0]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert response.parsed == {"label": "spam"}


@pytest.mark.parametrize(
    ("error", "kind", "retry_after"),
    [
        (
            openai.RateLimitError(
                "slow", response=_http_response(429, {"retry-after-ms": "2500"}), body=None
            ),
            ErrorKind.RATE_LIMITED,
            2.5,
        ),
        (
            openai.RateLimitError(
                "quota",
                response=_http_response(429),
                body={"error": {"code": "insufficient_quota"}},
            ),
            ErrorKind.QUOTA_EXHAUSTED,
            None,
        ),
        (
            openai.AuthenticationError("no", response=_http_response(401), body=None),
            ErrorKind.CLIENT,
            None,
        ),
        (
            openai.InternalServerError("boom", response=_http_response(503), body=None),
            ErrorKind.TRANSIENT,
            None,
        ),
        (
            openai.APITimeoutError(request=httpx2.Request("POST", "https://example.invalid")),
            ErrorKind.TIMEOUT,
            None,
        ),
        (
            openai.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid")),
            ErrorKind.CONNECTION,
            None,
        ),
    ],
)
def test_openai_errors_are_classified(
    error: Exception, kind: ErrorKind, retry_after: float | None
) -> None:
    client, _ = _openai_client(error)
    with pytest.raises(ProviderError) as info:
        asyncio.run(OpenAIModel(client=client).complete(REQUEST))
    assert info.value.kind is kind
    assert info.value.retry_after_s == retry_after
    assert info.value.provider == "openai"


def test_openai_builds_its_client_with_retries_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRE_TEST_OPENAI_KEY", "sk-dummy")
    model = OpenAIModel(api_key_env="AIRE_TEST_OPENAI_KEY", timeout_s=12.0)
    assert model.client.max_retries == 0
    assert model.server_address == "api.openai.com"
    asyncio.run(model.aclose())


def test_openai_without_a_key_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIRE_TEST_OPENAI_KEY", raising=False)
    with pytest.raises(ProviderError, match="AIRE_TEST_OPENAI_KEY"):
        OpenAIModel(api_key_env="AIRE_TEST_OPENAI_KEY")


# --- Anthropic ----------------------------------------------------------------


def _anthropic_client(result: Any) -> tuple[Any, Recorder]:
    create = Recorder(result)

    async def close() -> None:
        return None

    return SimpleNamespace(messages=SimpleNamespace(create=create), close=close), create


def _anthropic_message(text: str, stop_reason: str = "end_turn") -> AnthropicMessage:
    return AnthropicMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "m-2026",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 30,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 5,
            },
        }
    )


def test_anthropic_request_mapping_drops_temperature() -> None:
    client, create = _anthropic_client(_anthropic_message("hello"))
    response = asyncio.run(AnthropicModel(client=client).complete(REQUEST))
    (call,) = create.calls
    assert "temperature" not in call  # removed from messages.create in anthropic 2.x
    assert call["system"] == "Be brief."
    assert call["messages"] == [{"role": "user", "content": "Classify this."}]
    assert call["max_tokens"] == 64
    assert response.text == "hello"
    # Total input counts cache reads and writes too (GenAI semantic conventions).
    assert response.usage.input_tokens == 115
    assert response.usage.cache_read_tokens == 100
    assert response.usage.cache_write_tokens == 5
    assert response.usage.output_tokens == 30
    assert response.finish_reason == "end_turn"


def test_anthropic_structured_output_uses_output_config() -> None:
    client, create = _anthropic_client(_anthropic_message('{"label": "ham"}'))
    response = asyncio.run(AnthropicModel(client=client).complete(STRUCTURED))
    assert create.calls[0]["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert response.parsed == {"label": "ham"}


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (
            anthropic.RateLimitError(
                "slow", response=_http_response(429, {"retry-after": "4"}), body=None
            ),
            ErrorKind.RATE_LIMITED,
        ),
        (
            anthropic.OverloadedError("busy", response=_http_response(529), body=None),
            ErrorKind.TRANSIENT,
        ),
        (
            anthropic.BadRequestError("bad", response=_http_response(400), body=None),
            ErrorKind.CLIENT,
        ),
        (
            anthropic.BadRequestError(
                "billing",
                response=_http_response(400),
                body={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Your credit balance is too low",
                    }
                },
            ),
            ErrorKind.QUOTA_EXHAUSTED,
        ),
    ],
)
def test_anthropic_errors_are_classified(error: Exception, kind: ErrorKind) -> None:
    client, _ = _anthropic_client(error)
    with pytest.raises(ProviderError) as info:
        asyncio.run(AnthropicModel(client=client).complete(REQUEST))
    assert info.value.kind is kind


def test_anthropic_output_cut_off_before_any_text_is_malformed() -> None:
    client, _ = _anthropic_client(_anthropic_message("", stop_reason="max_tokens"))
    with pytest.raises(ProviderError) as info:
        asyncio.run(AnthropicModel(client=client).complete(REQUEST))
    assert info.value.kind is ErrorKind.MALFORMED


def test_anthropic_builds_its_client_with_retries_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRE_TEST_ANTHROPIC_KEY", "sk-ant-dummy")
    model = AnthropicModel(api_key_env="AIRE_TEST_ANTHROPIC_KEY")
    assert model.client.max_retries == 0
    asyncio.run(model.aclose())


# --- Gemini -------------------------------------------------------------------


def _gemini_client(result: Any) -> tuple[Any, Recorder]:
    generate = Recorder(result)

    async def aclose() -> None:
        return None

    aio = SimpleNamespace(models=SimpleNamespace(generate_content=generate), aclose=aclose)
    return SimpleNamespace(aio=aio), generate


def _gemini_response(text: str) -> Any:
    from google.genai import types

    return types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": text}]}, "finish_reason": "STOP"}
            ],
            "usage_metadata": {
                "prompt_token_count": 40,
                "candidates_token_count": 12,
                "thoughts_token_count": 7,
                "cached_content_token_count": 16,
                "total_token_count": 59,
            },
            "model_version": "gemini-test-001",
            "response_id": "resp-1",
        }
    )


def test_gemini_request_and_usage_mapping() -> None:
    client, generate = _gemini_client(_gemini_response("fine"))
    response = asyncio.run(GeminiModel(client=client).complete(REQUEST))
    (call,) = generate.calls
    config = call["config"]
    assert call["model"] == "m"
    assert config.system_instruction == "Be brief."
    assert config.temperature == 0.0
    assert config.max_output_tokens == 64
    assert [c.role for c in call["contents"]] == ["user"]
    assert response.text == "fine"
    assert response.response_model == "gemini-test-001"
    assert response.usage.input_tokens == 40
    assert response.usage.output_tokens == 19  # candidates plus thoughts
    assert response.usage.reasoning_tokens == 7
    assert response.usage.cache_read_tokens == 16
    assert response.finish_reason == "STOP"


def test_gemini_structured_output_is_requested_and_validated() -> None:
    client, generate = _gemini_client(_gemini_response('{"label": "x"}'))
    response = asyncio.run(GeminiModel(client=client).complete(STRUCTURED))
    config = generate.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == SCHEMA
    assert response.parsed == {"label": "x"}


@pytest.mark.parametrize(
    ("error", "kind", "retry_after"),
    [
        (
            genai_errors.ClientError(
                429,
                {
                    "error": {
                        "code": 429,
                        "status": "RESOURCE_EXHAUSTED",
                        "details": [{"retryDelay": "13s"}],
                    }
                },
            ),
            ErrorKind.RATE_LIMITED,
            13.0,
        ),
        (
            genai_errors.ClientError(
                429,
                {
                    "error": {
                        "code": 429,
                        "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}],
                    }
                },
            ),
            ErrorKind.QUOTA_EXHAUSTED,
            None,
        ),
        (genai_errors.ServerError(503, {"error": {"code": 503}}), ErrorKind.TRANSIENT, None),
        (genai_errors.ClientError(400, {"error": {"code": 400}}), ErrorKind.CLIENT, None),
        (httpx2.ReadTimeout("slow"), ErrorKind.TIMEOUT, None),
        (httpx2.ConnectError("refused"), ErrorKind.CONNECTION, None),
    ],
)
def test_gemini_errors_are_classified(
    error: Exception, kind: ErrorKind, retry_after: float | None
) -> None:
    client, _ = _gemini_client(error)
    with pytest.raises(ProviderError) as info:
        asyncio.run(GeminiModel(client=client).complete(REQUEST))
    assert info.value.kind is kind
    assert info.value.retry_after_s == retry_after
    assert info.value.provider == "gcp.gemini"


def test_gemini_http_options_leave_sdk_retries_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRE_TEST_GEMINI_KEY", "dummy")
    model = GeminiModel(api_key_env="AIRE_TEST_GEMINI_KEY", timeout_s=30.0, api_version="v1beta")
    assert model.http_options.retry_options is None  # google-genai: unset means one attempt
    assert model.http_options.timeout == 30_000  # milliseconds
    assert model.http_options.api_version == "v1beta"
    asyncio.run(model.aclose())


def test_an_unexpected_exception_is_not_disguised() -> None:
    client, _ = _openai_client(KeyError("bug in our code"))
    with pytest.raises(KeyError):
        asyncio.run(OpenAIModel(client=client).complete(REQUEST))


def test_gemini_defaults_to_the_v1beta_api(monkeypatch: pytest.MonkeyPatch) -> None:
    # Observed live on 2026-11-20: the v1 API answers system instructions with
    # HTTP 400 "Developer instruction is not enabled for api version v1".
    monkeypatch.setenv("AIRE_TEST_GEMINI_KEY", "dummy")
    model = GeminiModel(api_key_env="AIRE_TEST_GEMINI_KEY")
    assert model.http_options.api_version == "v1beta"
    asyncio.run(model.aclose())
