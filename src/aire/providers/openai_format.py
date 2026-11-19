"""The OpenAI chat-completions wire format, shared by the stdlib and SDK adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import ErrorKind, ProviderError
from .structured import parse_structured
from .types import ChatRequest, ChatResponse, TokenUsage


def chat_payload(request: ChatRequest) -> dict[str, Any]:
    """Keyword arguments (and JSON body) for chat.completions.create."""
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "max_tokens": request.max_output_tokens,
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.seed is not None:
        payload["seed"] = request.seed
    if request.json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": dict(request.json_schema),
                "strict": True,
            },
        }
    return payload


def _optional_int(container: Any, key: str) -> int | None:
    if isinstance(container, Mapping):
        value = container.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def usage_from_openai(usage: Any) -> TokenUsage:
    """Map an OpenAI usage block; cached and reasoning tokens are breakdowns, already included."""
    if not isinstance(usage, Mapping):
        return TokenUsage(source="none")
    return TokenUsage(
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        cache_read_tokens=_optional_int(usage.get("prompt_tokens_details"), "cached_tokens"),
        reasoning_tokens=_optional_int(usage.get("completion_tokens_details"), "reasoning_tokens"),
        source="api",
    )


def parse_chat_completion(data: Any, request: ChatRequest, *, provider: str) -> ChatResponse:
    """Turn a chat-completions response document into a ChatResponse."""
    try:
        choice = data["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ProviderError(
            f"malformed response: {type(exc).__name__}: {exc}",
            kind=ErrorKind.MALFORMED,
            provider=provider,
        ) from exc
    content = message.get("content") or ""
    finish_reason = choice.get("finish_reason") or ""
    if not content and finish_reason == "length":
        # A thinking model can spend the whole output cap before any content.
        raise ProviderError(
            "response truncated before any content (finish_reason=length)",
            kind=ErrorKind.MALFORMED,
            provider=provider,
        )
    parsed = (
        parse_structured(content, request.json_schema, provider=provider)
        if request.json_schema is not None
        else None
    )
    return ChatResponse(
        text=content,
        usage=usage_from_openai(data.get("usage")),
        response_model=str(data.get("model") or request.model),
        finish_reason=finish_reason,
        response_id=data.get("id"),
        parsed=parsed,
    )
