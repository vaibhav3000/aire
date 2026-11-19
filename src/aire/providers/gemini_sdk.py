"""GeminiModel: the google-genai 3.x SDK behind the ChatModel protocol.

Install with `pip install aire[gemini]`. Notes for this SDK version:

- The client is built without retry_options, which in google-genai means one
  attempt and no SDK retries; ModelClient retries (ADR 0001).
- 3.x defaults to the v1 API. Some features exist only on v1beta; pass
  api_version="v1beta" to use them.
- Structured output asks for application/json with response_json_schema;
  the SDK's own `parsed` is None on a mismatch without saying why, so aire
  validates the text itself.
- Output tokens are candidates plus thoughts; the prompt count already
  includes cached content.
"""

from __future__ import annotations

from typing import Any

from .sdk_errors import classify_sdk_exception, require_key
from .structured import parse_structured
from .types import ChatRequest, ChatResponse, TokenUsage

PROVIDER = "gcp.gemini"


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class GeminiModel:
    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        timeout_s: float = 60.0,
        api_version: str | None = None,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # the extra is not installed
            raise ImportError("GeminiModel needs google-genai: pip install 'aire[gemini]'") from exc
        self._types = types
        self.provider = PROVIDER
        self.server_address: str | None = "generativelanguage.googleapis.com"
        self.http_options = types.HttpOptions(
            timeout=int(timeout_s * 1000), api_version=api_version
        )
        if client is None:
            client = genai.Client(
                api_key=require_key(api_key_env, PROVIDER), http_options=self.http_options
            )
        self.client: Any = client

    def _config(self, request: ChatRequest) -> Any:
        types = self._types
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        fields: dict[str, Any] = {"max_output_tokens": request.max_output_tokens}
        if system:
            fields["system_instruction"] = system
        if request.temperature is not None:
            fields["temperature"] = request.temperature
        if request.seed is not None:
            fields["seed"] = request.seed
        if request.json_schema is not None:
            fields["response_mime_type"] = "application/json"
            fields["response_json_schema"] = dict(request.json_schema)
        return types.GenerateContentConfig(**fields)

    def _contents(self, request: ChatRequest) -> list[Any]:
        types = self._types
        return [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in request.messages
            if m.role != "system"
        ]

    async def complete(self, request: ChatRequest) -> ChatResponse:
        try:
            response = await self.client.aio.models.generate_content(
                model=request.model, contents=self._contents(request), config=self._config(request)
            )
        except Exception as exc:
            error = classify_sdk_exception(exc, provider=self.provider)
            if error is None:
                raise
            raise error from exc
        text = response.text or ""
        meta = response.usage_metadata
        candidates = response.candidates or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        finish_reason = str(getattr(finish, "name", finish or ""))
        if meta is None:
            usage = TokenUsage(source="none")
        else:
            thoughts = _int(meta.thoughts_token_count)
            usage = TokenUsage(
                input_tokens=_int(meta.prompt_token_count) or 0,
                output_tokens=(_int(meta.candidates_token_count) or 0) + (thoughts or 0),
                cache_read_tokens=_int(meta.cached_content_token_count),
                reasoning_tokens=thoughts,
                source="api",
            )
        parsed = (
            parse_structured(text, request.json_schema, provider=self.provider)
            if request.json_schema is not None
            else None
        )
        return ChatResponse(
            text=text,
            usage=usage,
            response_model=str(response.model_version or request.model),
            finish_reason=finish_reason,
            response_id=response.response_id,
            parsed=parsed,
        )

    async def aclose(self) -> None:
        await self.client.aio.aclose()
