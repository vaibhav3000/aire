"""LocalOpenAICompatModel: any OpenAI-style chat endpoint, with no dependencies.

Works with Ollama (http://localhost:11434/v1), vLLM (http://localhost:8000/v1),
the aire fake server, or a hosted OpenAI-compatible API. The key, when one is
needed, is read from the named environment variable at call time and is
never stored on the object.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from urllib.parse import urlsplit

from .errors import ErrorKind, ProviderError, classify_http_status
from .http import post_json
from .openai_format import chat_payload, parse_chat_completion
from .types import ChatRequest, ChatResponse


class LocalOpenAICompatModel:
    def __init__(
        self,
        base_url: str,
        *,
        provider: str = "openai",
        api_key_env: str | None = None,
        timeout_s: float = 60.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.server_address = urlsplit(self.base_url).hostname
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.extra_headers = dict(extra_headers or {})

    def _headers(self) -> dict[str, str]:
        headers = dict(self.extra_headers)
        if self.api_key_env is not None:
            key = os.environ.get(self.api_key_env, "").strip()
            if not key:
                raise ProviderError(
                    f"environment variable {self.api_key_env} is not set",
                    kind=ErrorKind.CLIENT,
                    provider=self.provider,
                )
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def complete(self, request: ChatRequest) -> ChatResponse:
        headers = self._headers()
        response = await post_json(
            f"{self.base_url}/chat/completions",
            chat_payload(request),
            headers=headers,
            timeout_s=self.timeout_s,
        )
        text = response.text()
        if response.status >= 400:
            raise classify_http_status(
                response.status, text, response.headers, provider=self.provider
            )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"response is not JSON: {exc}", kind=ErrorKind.MALFORMED, provider=self.provider
            ) from exc
        return parse_chat_completion(data, request, provider=self.provider)

    async def aclose(self) -> None:
        return None  # one connection per request: nothing stays open
