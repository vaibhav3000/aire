"""LLMRagSystem: local retrieval plus generation through a shared ModelClient.

Retrieval is AIRE's deterministic KeywordRetriever; generation goes through
the model client, so pacing, retries, caching and usage accounting are the
same for every provider. Each invocation reports:

- a retrieval span and one span per model attempt (failed ones included),
  which the runner hangs under the case's root span,
- the time spent waiting for the rate limiter and for backoff (wait_ms), so
  the runner keeps it out of latency,
- whether the answer was replayed from the response cache.
"""

from __future__ import annotations

import time
from typing import Any

from ..llm.base import SUTResult
from ..llm.deterministic import KeywordRetriever
from ..providers.client import AttemptRecord, Completion, ModelCallError, ModelClient
from ..providers.types import ChatRequest, Message
from ..runner.async_runner import AsyncSystemUnderTest
from ..schema import Span, new_id
from .prompts import RAG_SYSTEM_PROMPT, rag_user_prompt


def attempt_spans(
    attempts: list[AttemptRecord], *, provider: str, model: str, server_address: str | None
) -> list[Span]:
    """One generation span per model attempt, named like the GenAI convention."""
    spans = []
    for record in attempts:
        attributes: dict[str, Any] = {
            "operation": "chat",
            "provider": provider,
            "request_model": model,
            "attempt": record.attempt,
        }
        if server_address:
            attributes["server_address"] = server_address
        if record.backoff_s:
            attributes["backoff_s"] = round(record.backoff_s, 4)
        if record.queued_s:
            attributes["queued_s"] = round(record.queued_s, 4)
        if record.error_kind:
            attributes["error_kind"] = record.error_kind
        if record.status is not None:
            attributes["http_status"] = record.status
        spans.append(
            Span(
                kind="generation",
                name=f"chat {model}",
                started_at=record.started_at,
                latency_ms=round(record.latency_ms, 3),
                attributes=attributes,
                error=record.error,
                span_id=new_id("sp"),
            )
        )
    return spans


class LLMRagSystem(AsyncSystemUnderTest):
    def __init__(
        self,
        name: str,
        corpus: dict[str, str],
        client: ModelClient,
        model: str,
        *,
        top_k: int = 3,
        weighting: str = "idf",
        max_output_tokens: int = 256,
        temperature: float | None = 0.0,
        deterministic: bool = False,
    ) -> None:
        self.name = name
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.deterministic = deterministic
        self.retriever = KeywordRetriever(corpus, top_k=top_k, weighting=weighting)
        self.config = {
            "top_k": top_k,
            "weighting": weighting,
            "generator": f"{client.provider}:{model}",
            "temperature": "provider default" if temperature is None else temperature,
            "max_output_tokens": max_output_tokens,
        }

    async def aclose(self) -> None:
        await self.client.aclose()

    async def ainvoke(
        self, case_input: str, case_context: dict[str, Any] | None = None
    ) -> SUTResult:
        started_at = time.time()
        start = time.perf_counter()
        retrieved = self.retriever.retrieve(case_input)
        retrieval_span = Span(
            kind="retrieval",
            name="keyword_retriever.retrieve",
            started_at=started_at,
            latency_ms=round((time.perf_counter() - start) * 1000.0, 3),
            attributes={"top_k": self.config["top_k"], "n_results": len(retrieved)},
            span_id=new_id("sp"),
        )
        doc_ids = [r.doc_id for r in retrieved]
        user = rag_user_prompt(case_input, [(r.doc_id, r.text) for r in retrieved])
        request = ChatRequest(
            model=self.model,
            messages=(Message("system", RAG_SYSTEM_PROMPT), Message("user", user)),
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )
        server_address = getattr(self.client.model, "server_address", None)
        try:
            completion: Completion = await self.client.complete(request)
        except ModelCallError as exc:
            spans = attempt_spans(
                exc.attempts,
                provider=self.client.provider,
                model=self.model,
                server_address=server_address,
            )
            return SUTResult(
                answer=None,
                retrieved_doc_ids=doc_ids,
                prompt_text=user,
                error=f"ModelCallError: {exc}",
                wait_ms=exc.waited_s * 1000.0,
                span_attributes={
                    "provider_attempts": len(exc.attempts),
                    "error_kind": str(exc.kind),
                },
                spans=[retrieval_span, *spans],
            )
        response = completion.response
        usage = response.usage
        spans = attempt_spans(
            completion.attempts,
            provider=self.client.provider,
            model=self.model,
            server_address=server_address,
        )
        if spans:
            spans[-1].attributes.update(
                {
                    "response_model": response.response_model,
                    "finish_reason": response.finish_reason,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                }
            )
        attributes: dict[str, Any] = {
            "provider_attempts": len(completion.attempts),
            "response_model": response.response_model,
        }
        if completion.cache_hit:
            attributes["cache_hit"] = True
            attributes["cached_latency_ms"] = completion.cached_latency_ms
        return SUTResult(
            answer=response.text.strip(),
            retrieved_doc_ids=doc_ids,
            prompt_text=user,
            usage={
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "token_source": usage.source,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
            },
            wait_ms=completion.waited_s * 1000.0,
            span_attributes=attributes,
            spans=[retrieval_span, *spans],
        )
