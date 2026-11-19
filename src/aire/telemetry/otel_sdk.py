"""Bridge to the OpenTelemetry SDK (install with `pip install "aire[otel]"`).

- emit_traces() sends completed aire traces through a TracerProvider as
  GenAI spans with their recorded start and end times. Called inside an
  active span, the case spans become its children.
- JsonLinesSpanExporter writes whatever a provider exports as OTLP JSON
  lines, the same format aire.telemetry.otlp_json reads back.
- configure_tracer_provider() builds a provider that exports over OTLP/HTTP
  (protobuf), to a JSON lines file, or both.

The provider is passed in, never taken from or installed as the global one,
so evaluation code cannot change an application's tracing setup.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import SpanKind, Status, StatusCode

from .. import __version__
from ..schema import Trace
from .genai import SpanRecord, trace_to_spans
from .otlp_json import encode_records

_KINDS = {
    "INTERNAL": SpanKind.INTERNAL,
    "CLIENT": SpanKind.CLIENT,
    "SERVER": SpanKind.SERVER,
    "PRODUCER": SpanKind.PRODUCER,
    "CONSUMER": SpanKind.CONSUMER,
}


def emit_traces(
    traces: Iterable[Trace], provider: TracerProvider, *, capture_content: bool = False
) -> None:
    """Emit each trace's span tree through `provider` with the recorded timestamps."""
    tracer = provider.get_tracer("aire", __version__)
    for trace in traces:
        records = trace_to_spans(trace, capture_content=capture_content)
        opened: dict[str, otel_trace.Span] = {}
        for record in records:  # parents always come before their children
            parent = opened.get(record.parent_span_id or "")
            context = otel_trace.set_span_in_context(parent) if parent is not None else None
            span = tracer.start_span(
                record.name,
                context=context,
                kind=_KINDS[record.kind],
                attributes=record.attributes,
                start_time=record.start_ns,
            )
            if record.status == "ERROR":
                span.set_status(Status(StatusCode.ERROR, record.status_message))
            elif record.status == "OK":
                span.set_status(Status(StatusCode.OK))
            opened[record.span_id] = span
        for record in reversed(records):  # children end before their parents
            opened[record.span_id].end(end_time=record.end_ns)


def _plain(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def readable_to_record(span: ReadableSpan) -> SpanRecord:
    context = span.get_span_context()
    if context is None:
        raise ValueError(f"span {span.name!r} has no context")
    status = span.status.status_code.name if span.status is not None else "UNSET"
    return SpanRecord(
        trace_id=f"{context.trace_id:032x}",
        span_id=f"{context.span_id:016x}",
        parent_span_id=f"{span.parent.span_id:016x}" if span.parent is not None else None,
        name=span.name,
        kind=span.kind.name,
        start_ns=span.start_time or 0,
        end_ns=span.end_time or 0,
        attributes={k: _plain(v) for k, v in (span.attributes or {}).items()},
        status=status,
        status_message=(span.status.description or "") if span.status is not None else "",
    )


class JsonLinesSpanExporter(SpanExporter):
    """Appends one OTLP JSON document per export batch to a file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS
        resource = dict(spans[0].resource.attributes) if spans[0].resource else {}
        document = encode_records([readable_to_record(s) for s in spans], resource=resource)
        line = json.dumps(document) + "\n"
        try:
            with self._lock, open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
        except OSError:
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def configure_tracer_provider(
    *,
    service_name: str = "aire",
    otlp_endpoint: str | None = None,
    jsonl_path: str | Path | None = None,
) -> TracerProvider:
    """A new provider exporting to an OTLP/HTTP endpoint, a JSON lines file, or both."""
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if otlp_endpoint is not None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    if jsonl_path is not None:
        provider.add_span_processor(SimpleSpanProcessor(JsonLinesSpanExporter(jsonl_path)))
    return provider
