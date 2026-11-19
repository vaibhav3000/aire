"""ScriptedChatModel: an in-memory adapter for tests and offline demos.

Each call consumes the next script item: a ChatResponse is returned, a
ProviderError is raised, and a callable is called with the request and its
result handled the same way. Every request is recorded.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Union

from .errors import ProviderError
from .types import ChatRequest, ChatResponse

ScriptItem = Union[ChatResponse, ProviderError, Callable[[ChatRequest], ChatResponse]]  # noqa: UP007


class ScriptedChatModel:
    def __init__(
        self,
        script: Sequence[ScriptItem] = (),
        *,
        default: Callable[[ChatRequest], ChatResponse] | None = None,
        provider: str = "fake",
        server_address: str | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self._script = list(script)
        self._default = default
        self.provider = provider
        self.server_address = server_address
        self.delay_s = delay_s
        self.requests: list[ChatRequest] = []
        self.closed = False

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self._script:
            item = self._script.pop(0)
        elif self._default is not None:
            item = self._default
        else:
            raise AssertionError("ScriptedChatModel ran out of script items")
        if isinstance(item, ProviderError):
            raise item
        if isinstance(item, ChatResponse):
            return item
        return item(request)

    async def aclose(self) -> None:
        self.closed = True
