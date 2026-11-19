"""Disk cache of model responses, keyed by the exact request.

The key is the SHA-256 of canonical JSON holding the provider, the model,
the messages and every parameter that changes the output, plus the cache
format version. Transport settings (timeouts, retries) are not part of a
request, so they cannot split the cache. Any change to the prompt or the
parameters is a different key, so a hit is always an answer to exactly this
request (ADR 0006).

Layout: <directory>/<first two hex chars>/<key>.json, written atomically.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..tracing.storage import atomic_write_text
from .types import ChatRequest, ChatResponse

CACHE_FORMAT = 1


def request_key(provider: str, request: ChatRequest) -> str:
    """SHA-256 hex digest identifying one exact request to one provider."""
    document: dict[str, Any] = {
        "format": CACHE_FORMAT,
        "provider": provider,
        "model": request.model,
        "messages": [[m.role, m.content] for m in request.messages],
        "max_output_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "json_schema": request.json_schema,
        "seed": request.seed,
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedEntry:
    response: ChatResponse
    latency_ms: float  # how long the original call took


class ResponseCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> CachedEntry | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CachedEntry(
                response=ChatResponse.from_dict(data["response"]),
                latency_ms=float(data["latency_ms"]),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # A damaged entry must not stop a run; it is reported and recomputed.
            warnings.warn(
                f"unreadable cache entry {path}: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def put(self, key: str, response: ChatResponse, latency_ms: float) -> None:
        document = {"key": key, "latency_ms": round(latency_ms, 3), "response": response.to_dict()}
        atomic_write_text(self._path(key), json.dumps(document, indent=1, ensure_ascii=False))
