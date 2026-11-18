"""Evaluation engine: per-case scoring, aggregation and the failure taxonomy."""

import json
import os
import subprocess
import sys

from aire.evaluator.engine import EngineConfig, EvaluationEngine
from aire.evaluator.judges import GroundednessJudge
from aire.evaluator.metrics import metric_info
from aire.runner.dataset import EvalCase
from aire.schema import Span, Trace, Usage

CORPUS = {
    "doc_a": "The refund window is 30 days for annual plans. Credits are issued to the account.",
    "doc_b": "API rate limits are 60 requests per minute.",
}
REFUND = EvalCase(
    case_id="refund",
    input="How long is the refund window?",
    expected_keywords=["30 days"],
    supporting_doc_ids=["doc_a"],
)
LIMITS = EvalCase(
    case_id="limits",
    input="What are the API rate limits?",
    expected_keywords=["60 requests"],
    supporting_doc_ids=["doc_b"],
)
IPV6 = EvalCase(
    case_id="ipv6", input="IPv6-only clusters?", expect_abstention=True, must_cite=False
)
GOOD_REFUND = "The refund window is 30 days for annual plans [doc:doc_a]."
GOOD_LIMITS = "API rate limits are 60 requests per minute [doc:doc_b]."
ENGINE = EvaluationEngine(corpus=CORPUS)


def _trace(
    case_id: str,
    answer: str | None,
    *,
    retrieved: tuple[str, ...] = (),
    error: str | None = None,
    latency_ms: float = 1.0,
    usage: Usage | None = None,
    tool_calls: tuple[dict, ...] = (),
) -> Trace:
    return Trace(
        run_id="r",
        case_id=case_id,
        system_config={},
        final_answer=answer,
        retrieved_doc_ids=list(retrieved),
        tool_calls=list(tool_calls),
        spans=[Span(kind="generation", name="g", started_at=0.0, latency_ms=latency_ms)],
        usage=usage or Usage(prompt_tokens=10, completion_tokens=5, token_source="heuristic"),
        error=error,
    )


# ---------------------------------------------------------------- determinism

_AGGREGATE_SCRIPT = """
import json
from aire.evaluator.engine import EvaluationEngine
from aire.runner.dataset import EvalCase
from aire.schema import Trace

corpus = {"doc_a": "The refund window is 30 days for annual plans."}
cases = [
    EvalCase(case_id="c1", input="refund window?", expected_keywords=["30 days"],
             supporting_doc_ids=["doc_a"]),
    EvalCase(case_id="c2", input="ipv6?", expect_abstention=True, must_cite=False),
]
traces = [
    Trace(run_id="r", case_id="c1", system_config={}, retrieved_doc_ids=["doc_a"],
          final_answer="The refund window is 30 days for annual plans [doc:doc_a]."),
    Trace(run_id="r", case_id="c2", system_config={}, final_answer="I don't know."),
]
results = EvaluationEngine(corpus=corpus).evaluate_run(traces, cases)
print(json.dumps(results["aggregates"]))
"""


def _aggregates_with_hash_seed(seed: int) -> str:
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    proc = subprocess.run(
        [sys.executable, "-c", _AGGREGATE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return proc.stdout


def test_aggregates_serialise_identically_across_hash_seeds() -> None:
    """Committed eval_results.json must not churn with Python's string hash seed."""
    outputs = {_aggregates_with_hash_seed(seed) for seed in range(6)}
    assert len(outputs) == 1, outputs
    assert json.loads(outputs.pop())["n_cases"] == 2


# ---------------------------------------------------------------- failed invocations


def test_errored_unanswerable_case_is_not_a_correct_abstention() -> None:
    row = ENGINE.evaluate_case(_trace("ipv6", None, error="ProviderError: HTTP 429"), IPV6)
    assert row["abstention_correct"] == 0.0
    assert row["abstention_accuracy"] == 0.0
    assert row["errors"] == 1
    assert row["failure"] == "invocation_error"


def test_errored_answerable_case_scores_zero_on_quality_metrics() -> None:
    trace = _trace("refund", None, retrieved=("doc_a",), error="TimeoutError: timed out")
    row = ENGINE.evaluate_case(trace, REFUND)
    for metric in (
        "keyword_recall",
        "groundedness",
        "gold_citation_recall",
        "citation_precision",
        "abstention_accuracy",
    ):
        assert row[metric] == 0.0, metric
    assert row["false_refusal"] is None


def test_empty_answer_without_an_error_is_an_invocation_error() -> None:
    row = ENGINE.evaluate_case(_trace("ipv6", ""), IPV6)
    assert row["errors"] == 1
    assert row["failure"] == "invocation_error"


def test_success_rate_and_error_rate_are_aggregated() -> None:
    traces = [
        _trace("refund", GOOD_REFUND, retrieved=("doc_a",)),
        _trace("ipv6", None, error="boom"),
    ]
    agg = ENGINE.evaluate_run(traces, [REFUND, IPV6])["aggregates"]
    assert agg["success_rate"] == 0.5
    assert agg["mean_errors"] == 0.5


def test_every_mean_reports_how_many_cases_it_averages() -> None:
    traces = [
        _trace("refund", GOOD_REFUND, retrieved=("doc_a",)),
        _trace("limits", "I don't know.", retrieved=("doc_b",)),
    ]
    agg = ENGINE.evaluate_run(traces, [REFUND, LIMITS])["aggregates"]
    means = [key.removeprefix("mean_") for key in agg if key.startswith("mean_")]
    assert means
    assert [m for m in means if f"n_{m}" not in agg] == []
    # The refusal makes no claim, so groundedness averages one case, not two.
    assert agg["mean_groundedness"] == 1.0
    assert agg["n_groundedness"] == 1
    assert agg["n_keyword_recall"] == 2


def test_failed_cases_are_excluded_from_cost_and_latency_summaries() -> None:
    traces = [
        _trace(
            "refund",
            GOOD_REFUND,
            retrieved=("doc_a",),
            latency_ms=100.0,
            usage=Usage(prompt_tokens=50, completion_tokens=10, token_source="api"),
        ),
        _trace(
            "limits",
            GOOD_LIMITS,
            retrieved=("doc_b",),
            latency_ms=300.0,
            usage=Usage(prompt_tokens=70, completion_tokens=10, token_source="api"),
        ),
        _trace(
            "ipv6",
            None,
            error="ProviderError: HTTP 429",
            latency_ms=38000.0,
            usage=Usage(token_source="none"),
        ),
    ]
    agg = ENGINE.evaluate_run(traces, [REFUND, LIMITS, IPV6])["aggregates"]
    assert agg["mean_latency_ms"] == 200.0
    assert agg["n_latency_ms"] == 2
    assert agg["p50_latency_ms"] == 200.0
    assert agg["p95_latency_ms"] == 290.0
    assert agg["mean_total_tokens"] == 70.0
    assert agg["token_sources"] == ["api"]


# ---------------------------------------------------------------- retrieval


def test_retrieval_is_scored_at_fixed_cutoffs() -> None:
    case = EvalCase(case_id="multi", input="q", supporting_doc_ids=["doc_a", "doc_b"])
    trace = _trace("multi", "Some answer.", retrieved=("doc_x", "doc_a", "doc_b"))
    row = ENGINE.evaluate_case(trace, case)
    assert row["retrieval_recall_at_1"] == 0.0
    assert row["retrieval_recall_at_3"] == 1.0
    assert row["retrieval_mrr_at_3"] == 0.5
    assert "retrieval_recall" not in row


def test_empty_retrieval_scores_zero_instead_of_disappearing() -> None:
    row = ENGINE.evaluate_case(_trace("refund", GOOD_REFUND), REFUND)
    assert row["retrieval_recall_at_3"] == 0.0
    assert row["retrieval_mrr_at_3"] == 0.0


def test_duplicate_retrieved_ids_count_once() -> None:
    case = EvalCase(case_id="c", input="q", supporting_doc_ids=["doc_a"], must_cite=False)
    trace = _trace("c", "x", retrieved=("doc_x", "doc_x", "doc_a"))
    engine = EvaluationEngine(corpus=CORPUS, config=EngineConfig(recall_ks=(2,)))
    row = engine.evaluate_case(trace, case)
    assert row["retrieval_recall_at_2"] == 1.0
    assert row["retrieval_mrr_at_3"] == 0.5


# ---------------------------------------------------------------- failure taxonomy


def test_missing_gold_doc_is_a_retrieval_miss_even_when_the_answer_is_wrong() -> None:
    row = ENGINE.evaluate_case(_trace("refund", GOOD_LIMITS, retrieved=("doc_b",)), REFUND)
    assert row["keyword_recall"] == 0.0
    assert row["failure"] == "retrieval_miss"


def test_refusal_without_the_gold_doc_is_blamed_on_retrieval() -> None:
    row = ENGINE.evaluate_case(_trace("refund", "I don't know.", retrieved=("doc_b",)), REFUND)
    assert row["failure"] == "retrieval_miss"


def test_wrong_tool_is_a_tool_error() -> None:
    case = EvalCase(case_id="t", input="q", expected_tool="search", must_cite=False)
    trace = _trace("t", "done", tool_calls=({"name": "read", "args": {}, "valid": True},))
    assert ENGINE.evaluate_case(trace, case)["failure"] == "tool_error"


def test_wrong_exact_answer_is_an_answer_mismatch() -> None:
    case = EvalCase(case_id="e", input="2+2?", expected_answer="4", must_cite=False)
    assert ENGINE.evaluate_case(_trace("e", "5"), case)["failure"] == "answer_mismatch"


# ---------------------------------------------------------------- grounding and citations


def test_citing_a_document_that_was_not_retrieved_is_ungrounded() -> None:
    row = ENGINE.evaluate_case(_trace("refund", GOOD_REFUND, retrieved=("doc_b",)), REFUND)
    assert row["groundedness"] == 0.0


def test_groundedness_judge_uses_only_retrieved_documents() -> None:
    judge = GroundednessJudge(CORPUS)
    trace = _trace("refund", GOOD_REFUND, retrieved=("doc_b",))
    assert judge.score(trace, {})["score"] == 0.0


def test_groundedness_overlap_threshold_comes_from_engine_config() -> None:
    answer = (
        "The refund window is 30 days for annual plans and includes lifetime upgrades [doc:doc_a]."
    )
    trace = _trace("refund", answer, retrieved=("doc_a",))
    assert ENGINE.evaluate_case(trace, REFUND)["groundedness"] == 1.0
    strict = EvaluationEngine(corpus=CORPUS, config=EngineConfig(groundedness_min_overlap=0.9))
    assert strict.evaluate_case(trace, REFUND)["groundedness"] == 0.0


def test_citation_precision_counts_gold_and_retrieved_documents_as_allowed() -> None:
    answer = "The refund window is 30 days [doc:doc_a]. Limits are 60 per minute [doc:doc_c]."
    row = ENGINE.evaluate_case(_trace("refund", answer, retrieved=("doc_a", "doc_b")), REFUND)
    assert row["citation_precision"] == 0.5


# ---------------------------------------------------------------- abstention and registry


def test_false_refusals_lower_abstention_accuracy() -> None:
    traces = [
        _trace("refund", "I don't know.", retrieved=("doc_a",)),
        _trace("ipv6", "I don't know."),
    ]
    agg = ENGINE.evaluate_run(traces, [REFUND, IPV6])["aggregates"]
    assert agg["mean_abstention_correct"] == 1.0
    assert agg["mean_abstention_accuracy"] == 0.5
    assert agg["mean_false_refusal"] == 1.0


def test_every_emitted_metric_has_a_registered_direction() -> None:
    case = EvalCase(
        case_id="full",
        input="q",
        expected_keywords=["30 days"],
        expected_answer="x",
        supporting_doc_ids=["doc_a"],
        expected_tool="search",
        expected_tool_args={"q": "x"},
    )
    call = {"name": "search", "args": {"q": "x"}, "valid": True}
    trace = _trace("full", GOOD_REFUND, retrieved=("doc_a",), tool_calls=(call,))
    row = ENGINE.evaluate_case(trace, case)
    numeric = [k for k, v in row.items() if isinstance(v, int | float) and not isinstance(v, bool)]
    assert [k for k in numeric if metric_info(k) is None] == []
    assert metric_info("success_rate") is not None
