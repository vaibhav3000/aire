"""Calling models: one request type, one adapter per provider, one model client.

Importing this package loads no third-party SDK; the Gemini, OpenAI and
Anthropic adapters import their SDK only when constructed (extras
aire[gemini], aire[openai], aire[anthropic]).
"""

from .base import ChatModel
from .errors import ErrorKind, ProviderError
from .types import ChatRequest, ChatResponse, Message, TokenUsage

__all__ = [
    "ChatModel",
    "ChatRequest",
    "ChatResponse",
    "ErrorKind",
    "Message",
    "ProviderError",
    "TokenUsage",
]
