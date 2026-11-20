"""Build a ChatModel adapter from a provider name (used by the scripts)."""

from __future__ import annotations

from .base import ChatModel
from .local import LocalOpenAICompatModel

PROVIDERS = ("gemini", "openai", "anthropic", "local")


def provider_name(provider: str) -> str:
    """The gen_ai.provider.name each adapter reports (response cache keys use it)."""
    return {"gemini": "gcp.gemini", "openai": "openai", "anthropic": "anthropic"}.get(
        provider, "ollama"
    )


def sampling_temperature(provider: str) -> float | None:
    """0.0 where the provider accepts it; anthropic 2.x removed the parameter."""
    return None if provider == "anthropic" else 0.0


def build_chat_model(provider: str, base_url: str | None = None) -> ChatModel:
    """The adapter for a provider; SDK adapters import their SDK only here."""
    if provider == "gemini":
        from .gemini_sdk import GeminiModel

        return GeminiModel(api_key_env="GEMINI_API_KEY")
    if provider == "openai":
        from .openai_sdk import OpenAIModel

        return OpenAIModel(base_url=base_url)
    if provider == "anthropic":
        from .anthropic_sdk import AnthropicModel

        return AnthropicModel()
    if provider == "local":
        return LocalOpenAICompatModel(base_url or "http://localhost:11434/v1", provider="ollama")
    raise ValueError(f"unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}")
