"""Map a completed aire trace to OpenTelemetry spans that follow the GenAI conventions.

The GenAI semantic conventions are still in Development status (late 2026),
so every attribute name aire writes or reads is defined here and nowhere
else (ADR 0003). The span tree for one case:

    invoke_workflow <system>          INTERNAL, the whole case
      <system>.invoke                 INTERNAL, one per attempt (aire root spans)
        retrieval                     INTERNAL, operation "retrieval"
        chat <model>                  CLIENT, one per model attempt
      execute_tool <name>             INTERNAL, one per recorded tool call

Prompt and answer text are content that may hold personal data, so they are
only exported with capture_content=True (the conventions make them opt-in).
Ids are derived from the aire run and trace ids, so the same trace always
exports to the same span ids.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..schema import Span, Trace

# --- attribute names (GenAI semantic conventions, Development status) --------
OPERATION = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
AGENT_NAME = "gen_ai.agent.name"
CONVERSATION_ID = "gen_ai.conversation.id"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
FINISH_REASONS = "gen_ai.response.finish_reasons"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
CACHE_READ_TOKENS = "gen_ai.usage.cache_read.input_tokens"
OUTPUT_MESSAGES = "gen_ai.output.messages"
TOOL_NAME = "gen_ai.tool.name"
TOOL_TYPE = "gen_ai.tool.type"
TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
ERROR_TYPE = "error.type"
SERVER_ADDRESS = "server.address"

# --- aire's own attributes ---------------------------------------------------
AIRE_RUN_ID = "aire.run_id"
AIRE_CASE_ID = "aire.case_id"
AIRE_TRACE_ID = "aire.trace_id"
AIRE_RETRIEVED = "aire.retrieved_doc_ids"
AIRE_ERROR = "aire.error"
AIRE_PROMPT_TOKENS = "aire.usage.prompt_tokens"
AIRE_COMPLETION_TOKENS = "aire.usage.completion_tokens"
AIRE_TOKEN_SOURCE = "aire.usage.token_source"
AIRE_TOOL_VALID = "aire.tool.valid"
AIRE_TIMING = "aire.timing"
AIRE_SPAN_KIND = "aire.span.kind"


@dataclass
class SpanRecord:
    """One span, independent of any OpenTelemetry package."""

    trace_id: str  # 32 hex characters
    span_id: str  # 16 hex characters
    parent_span_id: str | None
    name: str
    kind: str  # "INTERNAL" | "CLIENT" | "SERVER" | "PRODUCER" | "CONSUMER"
    start_ns: int
    end_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "UNSET"  # "UNSET" | "OK" | "ERROR"
    status_message: str = ""


def _hex(seed: str, length: int) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:length]


def _ns(seconds: float) -> int:
    return round(seconds * 1_000_000_000)


def _system_name(trace: Trace) -> str:
    for span in trace.spans:
        if span.parent_id is None and span.name.endswith(".invoke"):
            return span.name.removesuffix(".invoke")
    return trace.run_id


def _span_attributes(span: Span) -> dict[str, Any]:
    """GenAI attributes for an aire child span, from the neutral keys aire records."""
    a = span.attributes
    if span.kind == "retrieval":
        out: dict[str, Any] = {OPERATION: "retrieval"}
        if "n_results" in a:
            out["aire.retrieval.n_results"] = a["n_results"]
        return out
    if span.kind != "generation" or "request_model" not in a:
        return {AIRE_SPAN_KIND: span.kind}
    out = {OPERATION: str(a.get("operation", "chat")), REQUEST_MODEL: str(a["request_model"])}
    mapping = {
        "provider": PROVIDER,
        "response_model": RESPONSE_MODEL,
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "server_address": SERVER_ADDRESS,
        "error_kind": ERROR_TYPE,
        "http_status": "http.response.status_code",
        "attempt": "aire.attempt",
    }
    for key, name in mapping.items():
        if key in a and a[key] is not None:
            out[name] = a[key]
    if a.get("finish_reason"):
        out[FINISH_REASONS] = [str(a["finish_reason"])]
    return out


def trace_to_spans(trace: Trace, *, capture_content: bool = False) -> list[SpanRecord]:
    """The span tree for one case, root first."""
    seed = f"{trace.run_id}/{trace.trace_id}"
    trace_id = _hex(seed, 32)
    start = _ns(trace.started_at)
    ends = [_ns(s.started_at + s.latency_ms / 1000.0) for s in trace.spans]
    end = max([start, *ends])
    system = _system_name(trace)
    root_id = _hex(f"{seed}/root", 16)

    root_attributes: dict[str, Any] = {
        OPERATION: "invoke_workflow",
        AGENT_NAME: system,
        AIRE_RUN_ID: trace.run_id,
        AIRE_CASE_ID: trace.case_id,
        AIRE_TRACE_ID: trace.trace_id,
        AIRE_RETRIEVED: list(trace.retrieved_doc_ids),
        AIRE_PROMPT_TOKENS: trace.usage.prompt_tokens,
        AIRE_COMPLETION_TOKENS: trace.usage.completion_tokens,
        AIRE_TOKEN_SOURCE: trace.usage.token_source,
    }
    if trace.usage.cache_read_tokens is not None:
        root_attributes[CACHE_READ_TOKENS] = trace.usage.cache_read_tokens
    if trace.error:
        root_attributes[AIRE_ERROR] = trace.error
    if capture_content and trace.final_answer is not None:
        root_attributes[OUTPUT_MESSAGES] = json.dumps(
            [{"role": "assistant", "parts": [{"type": "text", "content": trace.final_answer}]}]
        )
    records = [
        SpanRecord(
            trace_id=trace_id,
            span_id=root_id,
            parent_span_id=None,
            name=f"invoke_workflow {system}",
            kind="INTERNAL",
            start_ns=start,
            end_ns=end,
            attributes=root_attributes,
            status="ERROR" if trace.error else "OK",
            status_message=trace.error or "",
        )
    ]

    ids: dict[str, str] = {}
    for i, span in enumerate(trace.spans):
        ids[span.span_id or f"#{i}"] = _hex(f"{seed}/span/{i}", 16)
    for i, span in enumerate(trace.spans):
        own = ids[span.span_id or f"#{i}"]
        parent = ids.get(span.parent_id or "", root_id) if span.parent_id else root_id
        attributes = _span_attributes(span) if span.parent_id else {AIRE_SPAN_KIND: span.kind}
        is_chat = attributes.get(OPERATION) == "chat"
        span_start = _ns(span.started_at)
        records.append(
            SpanRecord(
                trace_id=trace_id,
                span_id=own,
                parent_span_id=parent,
                name=span.name,
                kind="CLIENT" if is_chat else "INTERNAL",
                start_ns=span_start,
                end_ns=span_start + round(span.latency_ms * 1_000_000),
                attributes=attributes,
                status="ERROR" if span.error else "OK",
                status_message=span.error or "",
            )
        )

    for i, call in enumerate(trace.tool_calls):
        name = str(call.get("name"))
        valid = call.get("valid", True) is not False
        attributes = {
            OPERATION: "execute_tool",
            TOOL_NAME: name,
            TOOL_TYPE: "function",
            AIRE_TOOL_VALID: valid,
            AIRE_TIMING: "not recorded",
        }
        if capture_content:
            attributes[TOOL_ARGUMENTS] = json.dumps(call.get("args", {}), sort_keys=True)
        error = call.get("error")
        records.append(
            SpanRecord(
                trace_id=trace_id,
                span_id=_hex(f"{seed}/tool/{i}", 16),
                parent_span_id=root_id,
                name=f"execute_tool {name}",
                kind="INTERNAL",
                start_ns=start,
                end_ns=start,
                attributes=attributes,
                status="OK" if valid and not error else "ERROR",
                status_message=str(error or ("rejected by the runtime" if not valid else "")),
            )
        )
    return records
