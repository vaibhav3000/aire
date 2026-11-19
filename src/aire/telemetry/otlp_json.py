"""OTLP JSON encoding and decoding, and ingestion of OpenTelemetry traces (stdlib only).

Writes the OTLP/JSON shape a collector's file exporter produces: one
`{"resourceSpans": [...]}` document per line, ids as hex strings, times as
decimal strings of Unix nanoseconds, enums as integers. Reads the same shape
back, from aire or from any application instrumented with the GenAI
conventions, and turns each trace into an aire Trace that the evaluation
engine can score:

- case id: aire.case_id, else gen_ai.conversation.id, else the trace id;
- answer: the text parts of gen_ai.output.messages on the root span;
- tool calls: execute_tool spans (gen_ai.tool.name and its arguments);
- usage: aire's own totals when present, else the sum of the inference
  spans' gen_ai.usage.input_tokens and gen_ai.usage.output_tokens;
- spans: every span with its real timing; the root span's duration is the
  case's latency.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..schema import Span, Trace, Usage
from ..tracing.storage import atomic_write_text
from . import genai
from .genai import SpanRecord

SCOPE_NAME = "aire"
_KINDS = {"INTERNAL": 1, "SERVER": 2, "CLIENT": 3, "PRODUCER": 4, "CONSUMER": 5}
_KIND_NAMES = {v: k for k, v in _KINDS.items()}
_STATUS = {"UNSET": 0, "OK": 1, "ERROR": 2}
_STATUS_NAMES = {v: k for k, v in _STATUS.items()}


def _value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, list | tuple):
        return {"arrayValue": {"values": [_value(v) for v in value]}}
    return {"stringValue": str(value)}


def _plain(value: Mapping[str, Any]) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "arrayValue" in value:
        return [_plain(v) for v in value["arrayValue"].get("values", [])]
    return None


def _attributes(items: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _value(v)} for k, v in items.items()]


def encode_records(
    records: Sequence[SpanRecord], resource: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """One OTLP JSON document holding `records`."""
    spans = []
    for r in records:
        span: dict[str, Any] = {
            "traceId": r.trace_id,
            "spanId": r.span_id,
            "name": r.name,
            "kind": _KINDS[r.kind],
            "startTimeUnixNano": str(r.start_ns),
            "endTimeUnixNano": str(r.end_ns),
            "attributes": _attributes(r.attributes),
            "status": {"code": _STATUS[r.status]},
        }
        if r.parent_span_id:
            span["parentSpanId"] = r.parent_span_id
        if r.status_message:
            span["status"]["message"] = r.status_message
        spans.append(span)
    from .. import __version__

    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attributes(resource or {"service.name": "aire"})},
                "scopeSpans": [
                    {"scope": {"name": SCOPE_NAME, "version": __version__}, "spans": spans}
                ],
            }
        ]
    }


def encode_traces(
    traces: Iterable[Trace], *, capture_content: bool = False
) -> list[dict[str, Any]]:
    """One OTLP JSON document per aire trace."""
    return [
        encode_records(genai.trace_to_spans(t, capture_content=capture_content)) for t in traces
    ]


def write_jsonl(
    traces: Iterable[Trace], path: str | Path, *, capture_content: bool = False
) -> Path:
    documents = encode_traces(traces, capture_content=capture_content)
    return atomic_write_text(path, "".join(json.dumps(d) + "\n" for d in documents))


def decode_documents(documents: Iterable[Mapping[str, Any]]) -> list[SpanRecord]:
    records = []
    for document in documents:
        for resource_spans in document.get("resourceSpans", []):
            for scope_spans in resource_spans.get("scopeSpans", []):
                for s in scope_spans.get("spans", []):
                    status = s.get("status") or {}
                    records.append(
                        SpanRecord(
                            trace_id=str(s["traceId"]),
                            span_id=str(s["spanId"]),
                            parent_span_id=s.get("parentSpanId") or None,
                            name=str(s.get("name", "")),
                            kind=_KIND_NAMES.get(int(s.get("kind", 1)), "INTERNAL"),
                            start_ns=int(s.get("startTimeUnixNano", 0)),
                            end_ns=int(s.get("endTimeUnixNano", 0)),
                            attributes={
                                a["key"]: _plain(a["value"]) for a in s.get("attributes", [])
                            },
                            status=_STATUS_NAMES.get(int(status.get("code", 0)), "UNSET"),
                            status_message=str(status.get("message", "")),
                        )
                    )
    return records


def read_jsonl(path: str | Path) -> list[SpanRecord]:
    """Span records from an OTLP JSON lines file (one document per line)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return decode_documents(json.loads(line) for line in lines if line.strip())


def _answer(messages: Any) -> str | None:
    if not isinstance(messages, str):
        return None
    try:
        parsed = json.loads(messages)
    except json.JSONDecodeError:
        return messages
    texts = [
        str(part.get("content", ""))
        for message in parsed
        if isinstance(message, Mapping)
        for part in message.get("parts", [])
        if isinstance(part, Mapping) and part.get("type", "text") == "text"
    ]
    return " ".join(texts) if texts else None


_SPAN_KINDS = {
    "chat": "generation",
    "generate_content": "generation",
    "text_completion": "generation",
    "retrieval": "retrieval",
    "execute_tool": "tool_call",
}


def spans_to_traces(records: Sequence[SpanRecord], *, run_id: str | None = None) -> list[Trace]:
    """Group span records by trace and turn each group into an aire Trace."""
    groups: dict[str, list[SpanRecord]] = {}
    for record in records:
        groups.setdefault(record.trace_id, []).append(record)
    traces = []
    for otel_trace_id, group in groups.items():
        ids = {r.span_id for r in group}
        roots = [r for r in group if not r.parent_span_id or r.parent_span_id not in ids]
        root = min(roots, key=lambda r: r.start_ns)
        a = root.attributes
        spans = []
        tool_calls: list[dict[str, Any]] = []
        input_tokens = output_tokens = 0
        saw_usage = False
        for r in sorted(group, key=lambda r: (r is not root, r.start_ns)):
            operation = str(r.attributes.get(genai.OPERATION, ""))
            parent = (
                None
                if r is root
                else (r.parent_span_id if r.parent_span_id in ids else root.span_id)
            )
            spans.append(
                Span(
                    kind=_SPAN_KINDS.get(operation, "custom"),
                    name=r.name,
                    started_at=r.start_ns / 1e9,
                    latency_ms=max(0.0, (r.end_ns - r.start_ns) / 1e6),
                    attributes=dict(r.attributes),
                    error=r.status_message or None if r.status == "ERROR" else None,
                    span_id=r.span_id,
                    parent_id=parent,
                )
            )
            if operation == "execute_tool":
                raw = r.attributes.get(genai.TOOL_ARGUMENTS)
                try:
                    args = json.loads(raw) if isinstance(raw, str) else {}
                except json.JSONDecodeError:
                    args = {"raw": raw}
                valid = r.attributes.get(genai.AIRE_TOOL_VALID)
                tool_calls.append(
                    {
                        "name": str(r.attributes.get(genai.TOOL_NAME, r.name)),
                        "args": args,
                        "valid": bool(valid) if valid is not None else r.status != "ERROR",
                        "error": r.status_message or None,
                    }
                )
            if genai.INPUT_TOKENS in r.attributes or genai.OUTPUT_TOKENS in r.attributes:
                saw_usage = True
                input_tokens += int(r.attributes.get(genai.INPUT_TOKENS) or 0)
                output_tokens += int(r.attributes.get(genai.OUTPUT_TOKENS) or 0)
        if genai.AIRE_PROMPT_TOKENS in a:
            usage = Usage(
                prompt_tokens=int(a[genai.AIRE_PROMPT_TOKENS]),
                completion_tokens=int(a.get(genai.AIRE_COMPLETION_TOKENS, 0)),
                token_source=str(a.get(genai.AIRE_TOKEN_SOURCE, "api")),
            )
        elif saw_usage:
            usage = Usage(
                prompt_tokens=input_tokens, completion_tokens=output_tokens, token_source="api"
            )
        else:
            usage = Usage(token_source="none")
        error = a.get(genai.AIRE_ERROR) or (
            root.status_message or "error" if root.status == "ERROR" else None
        )
        traces.append(
            Trace(
                run_id=run_id or str(a.get(genai.AIRE_RUN_ID, "otel_import")),
                case_id=str(
                    a.get(genai.AIRE_CASE_ID) or a.get(genai.CONVERSATION_ID) or otel_trace_id
                ),
                system_config={"source": "opentelemetry", "root_span": root.name},
                started_at=root.start_ns / 1e9,
                final_answer=_answer(a.get(genai.OUTPUT_MESSAGES)),
                retrieved_doc_ids=[str(x) for x in a.get(genai.AIRE_RETRIEVED, [])],
                tool_calls=tool_calls,
                spans=spans,
                usage=usage,
                error=str(error) if error else None,
                trace_id=str(a.get(genai.AIRE_TRACE_ID) or f"tr_{otel_trace_id[:8]}"),
            )
        )
    return traces
