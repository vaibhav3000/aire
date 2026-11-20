"""AnthropicModel: the official anthropic 2.x SDK behind the ChatModel protocol.

Install with `pip install aire[anthropic]`. Notes for this SDK version:

- messages.create no longer accepts temperature, top_p or top_k, so a
  request's temperature is not sent; the system under test records
  "provider default" sampling in its configuration.
- Structured output goes through output_config={"format": {"type":
  "json_schema", "schema": ...}} and is validated by aire.
- The client is built with max_retries=0 (ADR 0001).
- Total input tokens are input_tokens + cache reads + cache writes, as the
  OpenTelemetry GenAI conventions define them for Anthropic.
"""

from __future__ import annotations

from typing import Any

from .errors import ErrorKind, ProviderError
from .sdk_errors import classify_sdk_exception, require_key
from .structured import parse_structured
from .types import ChatRequest, ChatResponse, TokenUsage


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class AnthropicModel:
    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout_s: float = 60.0,
    ) -> None:
        self.provider = "anthropic"
        self.server_address: str | None = "api.anthropic.com"
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # the extra is not installed
                raise ImportError(
                    "AnthropicModel needs the anthropic SDK: pip install 'aire[anthropic]'"
                ) from exc
            client = anthropic.AsyncAnthropic(
                api_key=require_key(api_key_env, self.provider),
                timeout=timeout_s,
                max_retries=0,
            )
        self.client: Any = client

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_output_tokens,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
                if m.role != "system"
            ],
        }
        if system:
            payload["system"] = system
        if request.json_schema is not None:
            payload["output_config"] = {
                "format": {"type": "json_schema", "schema": dict(request.json_schema)}
            }
        return payload

    async def complete(self, request: ChatRequest) -> ChatResponse:
        try:
            message = await self.client.messages.create(**self._payload(request))
        except Exception as exc:
            error = classify_sdk_exception(exc, provider=self.provider)
            if error is None:
                raise
            raise error from exc
        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        stop_reason = str(message.stop_reason or "")
        if not text and stop_reason == "max_tokens":
            raise ProviderError(
                "response cut off before any text (stop_reason=max_tokens)",
                kind=ErrorKind.MALFORMED,
                provider=self.provider,
            )
        usage = message.usage
        cache_read = _int(getattr(usage, "cache_read_input_tokens", None))
        cache_write = _int(getattr(usage, "cache_creation_input_tokens", None))
        details = getattr(usage, "output_tokens_details", None)
        parsed = (
            parse_structured(text, request.json_schema, provider=self.provider)
            if request.json_schema is not None
            else None
        )
        return ChatResponse(
            text=text,
            usage=TokenUsage(
                input_tokens=int(usage.input_tokens) + (cache_read or 0) + (cache_write or 0),
                output_tokens=int(usage.output_tokens),
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                reasoning_tokens=_int(getattr(details, "thinking_tokens", None)),
                source="api",
            ),
            response_model=str(message.model),
            finish_reason=stop_reason,
            response_id=getattr(message, "id", None),
            parsed=parsed,
        )

    async def aclose(self) -> None:
        await self.client.close()
