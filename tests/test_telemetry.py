"""aire traces as OpenTelemetry GenAI spans: mapping, OTLP JSON, ingestion (ADR 0003)."""

import json
from pathlib import Path

from aire.schema import Span, Trace, Usage
from aire.telemetry import genai
from aire.telemetry.otlp_json import (
    decode_documents,
    encode_traces,
    read_jsonl,
    spans_to_traces,
    write_jsonl,
)


def _llm_trace() -> Trace:
    return Trace(
        run_id="run_1",
        case_id="refunds_window",
        system_config={"generator": "openai:m"},
        started_at=1000.0,
        final_answer="Refunds take 30 days [doc:doc_refunds].",
        retrieved_doc_ids=["doc_refunds", "doc_sla"],
        spans=[
            Span("generation", "rag.invoke", 1000.0, 900.0, span_id="sp_root"),
            Span(
                "retrieval",
                "keyword_retriever.retrieve",
                1000.0,
                2.0,
                parent_id="sp_root",
                span_id="sp_r",
            ),
            Span(
                "generation",
                "chat m",
                1000.01,
                300.0,
                attributes={
                    "provider": "openai",
                    "request_model": "m",
                    "error_kind": "rate_limited",
                    "http_status": 429,
                },
                error="HTTP 429: slow down",
                parent_id="sp_root",
                span_id="sp_a1",
            ),
            Span(
                "generation",
                "chat m",
                1000.5,
                400.0,
                attributes={
                    "provider": "openai",
                    "request_model": "m",
                    "response_model": "m-2026",
                    "finish_reason": "stop",
                    "input_tokens": 120,
                    "output_tokens": 12,
                    "server_address": "api.example.invalid",
                },
                parent_id="sp_root",
                span_id="sp_a2",
            ),
        ],
        usage=Usage(
            prompt_tokens=120, completion_tokens=12, token_source="api", cache_read_tokens=64
        ),
        trace_id="tr_abc",
    )


def _agent_trace() -> Trace:
    return Trace(
        run_id="agent",
        case_id="task_01",
        system_config={},
        started_at=0.0,
        final_answer="COMPLETE: changed a.py",
        tool_calls=[
            {"name": "search", "args": {"pattern": "x"}, "valid": True},
            {"name": "run_tests", "args": {"cmd": "pytest"}, "valid": False, "error": "rejected"},
        ],
        trace_id="tr_agent",
    )


def test_spans_follow_the_genai_conventions() -> None:
    records = genai.trace_to_spans(_llm_trace())
    root = records[0]
    assert root.parent_span_id is None
    assert root.attributes["gen_ai.operation.name"] == "invoke_workflow"
    assert root.attributes["aire.case_id"] == "refunds_window"
    chats = [r for r in records if r.attributes.get("gen_ai.operation.name") == "chat"]
    assert [c.name for c in chats] == ["chat m", "chat m"]
    failed, ok = chats
    assert failed.kind == "CLIENT"
    assert failed.status == "ERROR"
    assert failed.attributes["error.type"] == "rate_limited"
    assert ok.attributes["gen_ai.provider.name"] == "openai"
    assert ok.attributes["gen_ai.request.model"] == "m"
    assert ok.attributes["gen_ai.response.model"] == "m-2026"
    assert ok.attributes["gen_ai.response.finish_reasons"] == ["stop"]
    assert ok.attributes["gen_ai.usage.input_tokens"] == 120
    assert ok.attributes["gen_ai.usage.output_tokens"] == 12
    assert ok.attributes["server.address"] == "api.example.invalid"
    attempt = next(r for r in records if r.name == "rag.invoke")
    assert ok.parent_span_id == attempt.span_id
    assert attempt.parent_span_id == root.span_id
    assert ok.end_ns - ok.start_ns == 400_000_000


def test_content_is_only_exported_when_asked() -> None:
    without = genai.trace_to_spans(_llm_trace())[0].attributes
    assert "gen_ai.output.messages" not in without
    with_content = genai.trace_to_spans(_llm_trace(), capture_content=True)[0].attributes
    messages = json.loads(with_content["gen_ai.output.messages"])
    assert messages[0]["parts"][0]["content"].startswith("Refunds take 30 days")


def test_tool_calls_become_execute_tool_spans() -> None:
    records = genai.trace_to_spans(_agent_trace(), capture_content=True)
    tools = [r for r in records if r.attributes.get("gen_ai.operation.name") == "execute_tool"]
    assert [t.name for t in tools] == ["execute_tool search", "execute_tool run_tests"]
    assert tools[1].status == "ERROR"
    assert json.loads(tools[0].attributes["gen_ai.tool.call.arguments"]) == {"pattern": "x"}
    assert tools[0].attributes["aire.timing"] == "not recorded"


def test_ids_are_deterministic_and_well_formed() -> None:
    a = genai.trace_to_spans(_llm_trace())
    b = genai.trace_to_spans(_llm_trace())
    assert [r.span_id for r in a] == [r.span_id for r in b]
    assert len(a[0].trace_id) == 32 and len(a[0].span_id) == 16
    int(a[0].trace_id, 16)


def test_otlp_json_round_trip_keeps_what_evaluation_needs(tmp_path: Path) -> None:
    path = tmp_path / "spans.jsonl"
    write_jsonl([_llm_trace(), _agent_trace()], path, capture_content=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    document = json.loads(lines[0])
    span = document["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert isinstance(span["startTimeUnixNano"], str)
    assert span["kind"] in (1, 3)
    traces = spans_to_traces(read_jsonl(path))
    llm, agent = traces
    original = _llm_trace()
    assert (llm.run_id, llm.case_id, llm.final_answer) == (
        "run_1",
        "refunds_window",
        original.final_answer,
    )
    assert llm.retrieved_doc_ids == original.retrieved_doc_ids
    assert (llm.usage.prompt_tokens, llm.usage.completion_tokens, llm.usage.token_source) == (
        120,
        12,
        "api",
    )
    assert agent.tool_calls[0]["name"] == "search" and agent.tool_calls[0]["args"] == {
        "pattern": "x"
    }
    assert agent.tool_calls[1]["valid"] is False


def test_foreign_otlp_documents_are_ingested() -> None:
    """A trace from another instrumented app: no aire attributes at all."""
    document = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "shop-bot"}}]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "app"},
                        "spans": [
                            {
                                "traceId": "0af7651916cd43dd8448eb211c80319c",
                                "spanId": "b7ad6b7169203331",
                                "name": "invoke_agent shop",
                                "kind": 1,
                                "startTimeUnixNano": "1000000000",
                                "endTimeUnixNano": "3000000000",
                                "attributes": [
                                    {
                                        "key": "gen_ai.conversation.id",
                                        "value": {"stringValue": "conv-7"},
                                    },
                                    {
                                        "key": "gen_ai.output.messages",
                                        "value": {
                                            "stringValue": json.dumps(
                                                [
                                                    {
                                                        "role": "assistant",
                                                        "parts": [
                                                            {
                                                                "type": "text",
                                                                "content": "It ships in 2 days.",
                                                            }
                                                        ],
                                                    }
                                                ]
                                            )
                                        },
                                    },
                                ],
                                "status": {"code": 1},
                            },
                            {
                                "traceId": "0af7651916cd43dd8448eb211c80319c",
                                "spanId": "00f067aa0ba902b7",
                                "parentSpanId": "b7ad6b7169203331",
                                "name": "chat big-model",
                                "kind": 3,
                                "startTimeUnixNano": "1100000000",
                                "endTimeUnixNano": "2900000000",
                                "attributes": [
                                    {
                                        "key": "gen_ai.operation.name",
                                        "value": {"stringValue": "chat"},
                                    },
                                    {
                                        "key": "gen_ai.usage.input_tokens",
                                        "value": {"intValue": "900"},
                                    },
                                    {
                                        "key": "gen_ai.usage.output_tokens",
                                        "value": {"intValue": "40"},
                                    },
                                ],
                            },
                        ],
                    }
                ],
            }
        ]
    }
    (trace,) = spans_to_traces(decode_documents([document]), run_id="imported")
    assert trace.case_id == "conv-7"
    assert trace.run_id == "imported"
    assert trace.final_answer == "It ships in 2 days."
    assert (trace.usage.prompt_tokens, trace.usage.completion_tokens) == (900, 40)
    root = trace.spans[0]
    assert root.parent_id is None and root.latency_ms == 2000.0
    assert trace.spans[1].parent_id == root.span_id


def test_encode_traces_groups_spans_by_trace() -> None:
    documents = encode_traces([_llm_trace(), _agent_trace()])
    assert len(documents) == 2
    names = [s["name"] for s in documents[1]["resourceSpans"][0]["scopeSpans"][0]["spans"]]
    assert names[0].startswith("invoke_workflow")
