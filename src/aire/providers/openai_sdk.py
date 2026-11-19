"""OpenAIModel: the official openai 3.x SDK behind the ChatModel protocol.

Install with `pip install aire[openai]`. The SDK client is built with
max_retries=0, because retries belong to ModelClient (ADR 0001). Passing
base_url points the same adapter at any OpenAI-compatible server; for one
without an SDK installed, use LocalOpenAICompatModel.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from .openai_format import chat_payload, parse_chat_completion
from .sdk_errors import classify_sdk_exception, require_key
from .types import ChatRequest, ChatResponse


class OpenAIModel:
    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        timeout_s: float = 60.0,
        provider: str = "openai",
    ) -> None:
        self.provider = provider
        self.server_address = urlsplit(base_url).hostname if base_url else "api.openai.com"
        if client is None:
            try:
                import openai
            except ImportError as exc:  # the extra is not installed
                raise ImportError(
                    "OpenAIModel needs the openai SDK: pip install 'aire[openai]'"
                ) from exc
            client = openai.AsyncOpenAI(
                api_key=require_key(api_key_env, provider),
                base_url=base_url,
                timeout=timeout_s,
                max_retries=0,
            )
        self.client: Any = client

    async def complete(self, request: ChatRequest) -> ChatResponse:
        try:
            completion = await self.client.chat.completions.create(
                **chat_payload(request, max_tokens_field="max_completion_tokens")
            )
        except Exception as exc:
            error = classify_sdk_exception(exc, provider=self.provider)
            if error is None:
                raise
            raise error from exc
        data = completion.model_dump() if hasattr(completion, "model_dump") else completion
        return parse_chat_completion(data, request, provider=self.provider)

    async def aclose(self) -> None:
        await self.client.close()
