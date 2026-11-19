"""Span hierarchy and cache hits in the trace schema and the engine's latency."""

from aire.evaluator.engine import EvaluationEngine
from aire.runner.dataset import EvalCase
from aire.schema import Span, Trace, Usage

CASE = EvalCase(case_id="c1", input="q", expected_keywords=["30 days"], must_cite=False)


def _trace(spans: list[Span]) -> Trace:
    return Trace(
        run_id="r",
        case_id="c1",
        system_config={},
        final_answer="Refunds take 30 days.",
        spans=spans,
    )


def test_latency_sums_root_spans_only() -> None:
    spans = [
        Span("generation", "sut.invoke", 0.0, 120.0, span_id="s1"),
        Span("retrieval", "retrieve", 0.0, 5.0, parent_id="s1"),
        Span("generation", "chat m", 0.0, 100.0, parent_id="s1"),
    ]
    row = EvaluationEngine().evaluate_case(_trace(spans), CASE)
    assert row["latency_ms"] == 120.0


def test_a_cache_hit_has_no_latency_and_is_counted() -> None:
    cached = _trace([Span("generation", "sut.invoke", 0.0, 0.4, attributes={"cache_hit": True})])
    fresh = Trace(
        run_id="r",
        case_id="c2",
        system_config={},
        final_answer="Refunds take 30 days.",
        spans=[Span("generation", "sut.invoke", 0.0, 900.0)],
    )
    cases = [
        CASE,
        EvalCase(case_id="c2", input="q", expected_keywords=["30 days"], must_cite=False),
    ]
    results = EvaluationEngine().evaluate_run([cached, fresh], cases)
    rows = {r["case_id"]: r for r in results["per_case"]}
    assert rows["c1"]["latency_ms"] is None
    assert rows["c1"]["cache_hit"] is True
    assert results["aggregates"]["mean_latency_ms"] == 900.0
    assert results["aggregates"]["n_latency_ms"] == 1
    assert results["aggregates"]["n_cache_hits"] == 1


def test_span_ids_round_trip_and_are_omitted_when_unset() -> None:
    trace = _trace(
        [
            Span("generation", "a", 0.0, 1.0),
            Span("tool_call", "b", 0.0, 1.0, span_id="x", parent_id="y"),
        ]
    )
    data = trace.to_dict()
    assert "span_id" not in data["spans"][0]
    assert data["spans"][1]["span_id"] == "x"
    assert Trace.from_dict(data).spans[1].parent_id == "y"


def test_usage_breakdowns_round_trip() -> None:
    trace = _trace([])
    trace.usage = Usage(
        prompt_tokens=10, completion_tokens=2, token_source="api", cache_read_tokens=4
    )
    data = trace.to_dict()
    assert data["usage"]["cache_read_tokens"] == 4
    assert Trace.from_dict(data).usage.cache_read_tokens == 4
    assert data["schema_version"] == "1.2"
