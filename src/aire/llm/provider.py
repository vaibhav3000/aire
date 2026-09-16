"""Minimal OpenAI-compatible chat client for AIRE (stdlib only).

Shared design with repo-engineer's llm_provider: one provider class for any
OpenAI-style endpoint, API key read from the environment at call time and
never persisted.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    finish_reason: str


class ProviderError(RuntimeError):
    pass


class OpenAICompatProvider:
    """Chat-completions client for OpenAI-compatible endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        model: str,
        temperature: float = 0.0,
        timeout_s: float = 90.0,
        max_tokens: int = 1024,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ProviderError(
                f"environment variable {self.api_key_env} is not set; no API key "
                "available. Deterministic mode needs no key."
            )
        return key

    def chat(self, system: str, user: str, retries: int = 3) -> LLMResponse:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._key()}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read())
                choice = data["choices"][0]
                message = choice.get("message", {})
                usage = data.get("usage", {})
                content = message.get("content")
                if not content and choice.get("finish_reason") == "length":
                    # Thinking models can exhaust max_tokens before emitting
                    # content; retrying without progress would waste quota.
                    raise ProviderError("response truncated before content (finish_reason=length)")
                return LLMResponse(
                    content=content or "",
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    model=data.get("model", self.model),
                    finish_reason=choice.get("finish_reason", ""),
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last_error = ProviderError(f"HTTP {exc.code}: {detail}")
                if exc.code in (429, 500, 502, 503) and attempt < retries:
                    time.sleep(6.0 * (attempt + 1))  # rate-limit backoff
                    continue
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                last_error = ProviderError(f"{type(exc).__name__}: {exc}")
                if attempt < retries:
                    time.sleep(3.0)
                    continue
        raise last_error if last_error else ProviderError("request failed")
