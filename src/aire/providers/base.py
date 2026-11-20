"""The ChatModel protocol every provider adapter satisfies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import ChatRequest, ChatResponse


@runtime_checkable
class ChatModel(Protocol):
    """One provider, one attempt per call.

    provider is the OpenTelemetry gen_ai.provider.name value ("openai",
    "anthropic", "gcp.gemini", "ollama", ...). complete() makes exactly one
    attempt and raises aire.providers.errors.ProviderError on failure;
    retries, pacing and caching belong to aire.providers.client.ModelClient.
    """

    provider: str
    server_address: str | None

    async def complete(self, request: ChatRequest) -> ChatResponse: ...

    async def aclose(self) -> None: ...
