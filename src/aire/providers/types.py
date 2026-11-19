"""Provider-neutral request and response types.

Every adapter translates a ChatRequest into its provider's call and the
provider's reply into a ChatResponse, so the model client, the cache, the
systems under test and the telemetry mapping only ever see these types.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]
_ROLES = frozenset({"system", "user", "assistant"})
TokenSource = Literal["api", "estimate", "none"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"unknown message role {self.role!r}")


@dataclass(frozen=True)
class ChatRequest:
    """One chat completion request.

    temperature None means "the provider's default sampling"; adapters for
    providers that no longer accept the parameter omit it. json_schema asks
    for structured output that must validate against the schema.
    """

    model: str
    messages: tuple[Message, ...]
    max_output_tokens: int = 1024
    temperature: float | None = 0.0
    json_schema: Mapping[str, Any] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a chat request needs at least one message")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

    def estimated_tokens(self) -> int:
        """Upper-bound token cost used for rate budgeting before the call.

        About four characters per prompt token plus the full output cap; the
        model client reconciles it with the real usage afterwards.
        """
        prompt_chars = sum(len(m.content) for m in self.messages)
        return prompt_chars // 4 + self.max_output_tokens


@dataclass(frozen=True)
class TokenUsage:
    """Token counts as the provider reported them.

    input_tokens includes cached prompt tokens; output_tokens includes
    reasoning tokens. The breakdowns are None when the provider did not
    report them.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    source: TokenSource = "api"


@dataclass(frozen=True)
class ChatResponse:
    text: str
    usage: TokenUsage
    response_model: str
    finish_reason: str
    response_id: str | None = None
    parsed: Any = None  # the validated JSON value when the request had a json_schema
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["extra"] = dict(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChatResponse:
        usage = TokenUsage(**dict(data.get("usage") or {}))
        return cls(
            text=str(data["text"]),
            usage=usage,
            response_model=str(data.get("response_model", "")),
            finish_reason=str(data.get("finish_reason", "")),
            response_id=data.get("response_id"),
            parsed=data.get("parsed"),
            extra=dict(data.get("extra") or {}),
        )
