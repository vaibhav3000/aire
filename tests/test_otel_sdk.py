"""The OpenTelemetry SDK bridge (aire[otel]): emitted spans, JSON lines and OTLP export.

Each test uses its own TracerProvider; nothing touches the global provider,
and the OTLP endpoint is a local HTTP server on 127.0.0.1.
"""

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from aire.schema import Span, Trace, Usage
from aire.telemetry.otel_sdk import JsonLinesSpanExporter, configure_tracer_provider, emit_traces
from aire.telemetry.otlp_json import read_jsonl, spans_to_traces


def _trace(case_id: str = "c1") -> Trace:
    return Trace(
        run_id="run_1",
        case_id=case_id,
        system_config={},
        started_at=2000.0,
        final_answer="Thirty days [doc:a].",
        spans=[
            Span("generation", "rag.invoke", 2000.0, 500.0, span_id="root"),
            Span(
                "generation",
                "chat m",
                2000.1,
                300.0,
                attributes={
                    "provider": "openai",
                    "request_model": "m",
                    "input_tokens": 50,
                    "output_tokens": 7,
                },
                parent_id="root",
                span_id="a1",
            ),
        ],
        usage=Usage(prompt_tokens=50, completion_tokens=7, token_source="api"),
        trace_id=f"tr_{case_id}",
    )


def test_emitted_spans_keep_the_tree_the_times_and_the_attributes() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    emit_traces([_trace()], provider)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    root, attempt, chat = spans["invoke_workflow rag"], spans["rag.invoke"], spans["chat m"]
    root_ctx, attempt_ctx = root.get_span_context(), attempt.get_span_context()
    assert root_ctx is not None and attempt_ctx is not None
    assert root.parent is None
    assert attempt.parent is not None and attempt.parent.span_id == root_ctx.span_id
    assert chat.parent is not None and chat.parent.span_id == attempt_ctx.span_id
    assert chat.kind.name == "CLIENT"
    assert chat.attributes is not None and chat.attributes["gen_ai.usage.input_tokens"] == 50
    assert chat.start_time == 2_000_100_000_000
    assert chat.end_time == 2_000_400_000_000
    contexts = [s.get_span_context() for s in spans.values()]
    assert len({c.trace_id for c in contexts if c is not None}) == 1
    assert None not in contexts


def test_the_json_lines_exporter_writes_ingestible_otlp(tmp_path: Path) -> None:
    path = tmp_path / "spans.jsonl"
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(JsonLinesSpanExporter(path)))
    emit_traces([_trace("c1"), _trace("c2")], provider, capture_content=True)
    provider.shutdown()
    traces = spans_to_traces(read_jsonl(path))
    assert sorted(t.case_id for t in traces) == ["c1", "c2"]
    assert traces[0].final_answer == "Thirty days [doc:a]."
    assert (traces[0].usage.prompt_tokens, traces[0].usage.completion_tokens) == (50, 7)


class _Collector:
    """A stand-in OTLP/HTTP collector that keeps every request body."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, bytes]] = []
        collector = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                collector.requests.append((self.path, self.headers.get("Content-Type", ""), body))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1/traces"


@pytest.fixture
def collector() -> Iterator[_Collector]:
    c = _Collector()
    c.thread.start()
    yield c
    c.server.shutdown()
    c.server.server_close()


def test_otlp_export_reaches_a_collector_as_protobuf(collector: _Collector, tmp_path: Path) -> None:
    provider = configure_tracer_provider(
        service_name="aire-test", otlp_endpoint=collector.endpoint, jsonl_path=tmp_path / "s.jsonl"
    )
    emit_traces([_trace()], provider)
    assert provider.force_flush()
    provider.shutdown()
    assert collector.requests, "the collector received nothing"
    path, content_type, body = collector.requests[0]
    assert path == "/v1/traces"
    assert content_type == "application/x-protobuf"
    request = ExportTraceServiceRequest()
    request.ParseFromString(body)
    names = {s.name for rs in request.resource_spans for ss in rs.scope_spans for s in ss.spans}
    assert {"invoke_workflow rag", "rag.invoke", "chat m"} <= names
    service = {
        a.key: a.value.string_value for rs in request.resource_spans for a in rs.resource.attributes
    }
    assert service["service.name"] == "aire-test"
    assert (tmp_path / "s.jsonl").read_text(encoding="utf-8").strip()
