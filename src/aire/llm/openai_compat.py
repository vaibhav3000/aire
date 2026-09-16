"""Real-LLM responder for AIRE: OpenAI-compatible chat generation over
AIRE's retriever.

The retrieval step stays local and deterministic (KeywordRetriever); the
generation step calls the configured provider. The system prompt requires
citations in AIRE's [doc:<id>] format and abstention when the context does
not contain the answer - so the deterministic groundedness judge and failure
taxonomy apply to real LLM answers unchanged.

The API key is read from an environment variable at call time and is never
written to traces, results or logs.
"""

from __future__ import annotations

import time
from typing import Any

from ..llm.base import SUTResult
from .deterministic import KeywordRetriever
from .provider import LLMResponse, OpenAICompatProvider

_SYSTEM_PROMPT = (
    "You are a customer-support assistant for Northwind Cloud. Answer the "
    "user's question using ONLY the documents provided in the context. "
    "Rules: (1) End every factual sentence with a citation in the exact form "
    "[doc:<document_id>] using the ids given. (2) If the documents do not "
    "contain the answer, reply with exactly: I don't know. (3) Do not invent "
    "policies, numbers or features. (4) Answer in at most three sentences."
)


class OpenAIChatResponder:
    """SystemUnderTest whose generation step is a real LLM.

    deterministic = False: the runner reports this in the manifest so results
    are never confused with the reproducible deterministic mode.
    """

    def __init__(
        self,
        name: str,
        corpus: dict[str, str],
        provider: OpenAICompatProvider,
        top_k: int = 3,
        min_score: float = 0.05,
        weighting: str = "idf",
        min_call_interval_s: float = 0.0,
    ) -> None:
        self.name = name
        self.corpus = corpus
        self.config = {
            "top_k": top_k,
            "min_score": min_score,
            "weighting": weighting,
            "generator": f"{provider.model} via OpenAI-compatible API",
            "temperature": provider.temperature,
        }
        self.retriever = KeywordRetriever(corpus, top_k=top_k, weighting=weighting)
        self.provider = provider
        self.min_call_interval_s = min_call_interval_s
        self._last_call_start = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def _pace(self) -> None:
        """Sleep so calls respect the provider's rate limit; runs BEFORE the
        runner's timer starts, so measured latency is the system's, not the
        rate limiter's."""
        if self.min_call_interval_s <= 0:
            return
        elapsed = time.time() - self._last_call_start
        if elapsed < self.min_call_interval_s:
            time.sleep(self.min_call_interval_s - elapsed)

    def _build_context(self, query: str) -> tuple[str, list[str]]:
        retrieved = self.retriever.retrieve(query)
        doc_ids = [r.doc_id for r in retrieved]
        parts = []
        for r in retrieved:
            parts.append(f"[doc:{r.doc_id}]\n{r.text}")
        context = "\n\n".join(parts) if parts else "(no documents found)"
        return context, doc_ids

    def invoke(self, case_input: str, case_context: dict[str, Any] | None = None) -> SUTResult:
        self._pace()
        self._last_call_start = time.time()
        context, doc_ids = self._build_context(case_input)
        user = f"Context documents:\n{context}\n\nQuestion: {case_input}"
        try:
            resp: LLMResponse = self.provider.chat(_SYSTEM_PROMPT, user)
        except Exception as exc:  # noqa: BLE001 - recorded as trace error
            return SUTResult(answer=None, retrieved_doc_ids=doc_ids,
                             prompt_text=user, error=f"{type(exc).__name__}: {exc}")
        self.calls += 1
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens
        return SUTResult(
            answer=resp.content.strip(),
            retrieved_doc_ids=doc_ids,
            prompt_text=user,
            usage={
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "token_source": "api",
                "model": resp.model,
            },
        )
