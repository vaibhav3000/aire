"""AIRE test suite: metrics, judges, tracing, runner, engine, regression, report."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aire.evaluator import metrics as M  # noqa: E402
from aire.evaluator import retrieval_metrics as R  # noqa: E402
from aire.evaluator import tool_metrics as T  # noqa: E402
from aire.evaluator.engine import EvaluationEngine, classify_failure  # noqa: E402
from aire.llm.deterministic import ExtractiveRAG, KeywordRetriever  # noqa: E402
from aire.regression import RegressionThresholds, compare_runs  # noqa: E402
from aire.runner.dataset import EvalCase, load_eval_dataset  # noqa: E402
from aire.runner.runner import ExperimentRunner, RunnerConfig  # noqa: E402
from aire.schema import Span, Trace, Usage, validate_trace_dict  # noqa: E402
from aire.tracing.recorder import TraceRecorder  # noqa: E402
from aire.tracing.storage import RunStore, load_run_traces  # noqa: E402

CORPUS = {
    "doc_a": "The refund window is 30 days for annual plans. Credits are issued to the account.",
    "doc_b": "API rate limits are 60 requests per minute. A 429 response means retry after backoff.",
}


# ---------------------------------------------------------------- metrics


def test_keyword_recall_counts_hits() -> None:
    ans = "The refund window is 30 days."
    assert M.keyword_recall(ans, ["refund", "30 days", "missing"]) == pytest.approx(2 / 3)
    assert M.keyword_recall(ans, []) == 1.0
    assert M.keyword_recall(None, ["x"]) == 0.0


def test_citation_parsing_and_coverage() -> None:
    ans = "Rate limits are 60 rpm [doc:doc_b]. Also see [doc:doc_z]."
    assert M.citation_ids(ans) == ["doc_b", "doc_z"]
    assert M.citation_coverage(ans, ["doc_b"]) == 1.0
    assert M.citation_coverage(ans, ["doc_b", "doc_a"]) == 0.5


def test_groundedness_supported_vs_unsupported() -> None:
    good = "Credits are issued to the account [doc:doc_a]."
    bad = "We offer telepathic support [doc:doc_a]."
    uncited = "The refund window is 30 days."
    assert M.groundedness(good, CORPUS) == 1.0
    assert M.groundedness(bad, CORPUS) == 0.0
    assert M.groundedness(uncited, CORPUS) == 0.0
    assert M.groundedness("I don't know.", CORPUS) is None


def test_retrieval_metrics() -> None:
    assert R.retrieval_recall_at_k(["a", "b", "c"], ["b", "c"]) == 1.0
    assert R.retrieval_recall_at_k(["a", "b", "c"], ["b", "c"], k=1) == 0.0
    assert R.retrieval_recall_at_k([], []) == 1.0
    assert R.retrieval_mrr(["x", "b"], ["b"]) == 0.5
    assert R.retrieval_mrr(["x", "y"], ["b"]) == 0.0


def test_tool_metrics() -> None:
    calls = [{"name": "search", "args": {"q": "x"}, "valid": True},
             {"name": "read", "args": {}, "valid": False}]
    assert T.tool_selection_correct(calls, "search") == 1.0
    assert T.tool_selection_correct(calls, "write") == 0.0
    assert T.tool_selection_correct(calls, None) is None
    assert T.tool_args_valid(calls) == 0.5
    assert T.tool_args_valid([]) == 1.0
    assert T.tool_args_match(calls, "search", {"q": "x"}) == 1.0
    assert T.tool_args_match(calls, "nope", {"q": "x"}) == 0.0


# ---------------------------------------------------------------- schema/tracing


def _trace_dict(**overrides):
    d = Trace(run_id="r1", case_id="c1", system_config={}).to_dict()
    d.update(overrides)
    return d


def test_trace_validation() -> None:
    assert validate_trace_dict(_trace_dict()) == []
    problems = validate_trace_dict({"spans": [{"kind": "retrieval"}]})
    assert any("case_id" in p for p in problems)
    assert any("latency_ms" in p for p in problems)


def test_trace_round_trip() -> None:
    t = Trace(run_id="r", case_id="c", system_config={"v": 1})
    t.spans.append(Span(kind="generation", name="g", started_at=0.0, latency_ms=1.5))
    t.usage = Usage(prompt_tokens=3, completion_tokens=4, token_source="heuristic")
    t2 = Trace.from_dict(t.to_dict())
    assert t2 == t


def test_recorder_and_store(tmp_path: Path) -> None:
    rec = TraceRecorder("run1", "case1", {"top_k": 2})
    with rec.span("retrieval", "kw") as span:
        span.attributes["k"] = 2
    trace = rec.finish("answer [doc:doc_a]", retrieved_doc_ids=["doc_a"])
    assert trace.spans[0].kind == "retrieval" and trace.spans[0].latency_ms >= 0

    store = RunStore(tmp_path, "run1", "demo")
    dummy = tmp_path / "ds.jsonl"
    dummy.write_text("line\n")
    store.write_manifest(0.0, {}, dummy, 1)
    store.append_trace(trace)
    store.flush_traces()
    loaded = load_run_traces(tmp_path / "run1")
    assert loaded[0].case_id == "case1"


def test_store_rejects_invalid_trace(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "runX", "demo")
    bad = Trace(run_id="", case_id="c", system_config={})
    with pytest.raises(ValueError):
        store.append_trace(bad)


# ---------------------------------------------------------------- SUT + runner


def test_keyword_retriever_ranks_by_overlap() -> None:
    r = KeywordRetriever(CORPUS, top_k=2)
    docs = r.retrieve("refund window annual plans")
    assert docs[0].doc_id == "doc_a"
    assert r.retrieve("zzz qqq") == []  # no signal -> no results


def test_extractive_rag_faithful_cites() -> None:
    sut = ExtractiveRAG("t", CORPUS, top_k=1, min_score=0.05)
    out = sut.invoke("What is the refund window for annual plans?")
    assert "[doc:doc_a]" in out.answer
    assert out.retrieved_doc_ids == ["doc_a"]


def test_extractive_rag_fluent_drops_citations() -> None:
    sut = ExtractiveRAG("t", CORPUS, top_k=1, min_score=0.05, style="fluent")
    out = sut.invoke("What is the refund window for annual plans?")
    assert "[doc:" not in out.answer and out.answer.strip()


def test_extractive_rag_abstains_on_unknown() -> None:
    sut = ExtractiveRAG("t", CORPUS, top_k=1, min_score=0.5)
    out = sut.invoke("What is the swahili phone support policy?")
    assert out.answer == "I don't know."


def test_runner_persists_traces(tmp_path: Path) -> None:
    sut = ExtractiveRAG("t", CORPUS, top_k=1)
    ds = tmp_path / "ds.jsonl"
    ds.write_text(json.dumps([
        {"case_id": "c1", "input": "What is the refund window for annual plans?",
         "expected_keywords": ["30 days"], "supporting_doc_ids": ["doc_a"]},
        {"case_id": "c2", "input": "Do you offer swahili phone support?",
         "expect_abstention": True, "must_cite": False},
    ]) + "\n")
    # load_eval_dataset expects JSONL lines
    ds.write_text(
        json.dumps({"case_id": "c1", "input": "What is the refund window for annual plans?",
                    "expected_keywords": ["30 days"], "supporting_doc_ids": ["doc_a"]}) + "\n" +
        json.dumps({"case_id": "c2", "input": "Do you offer swahili phone support?",
                    "expect_abstention": True, "must_cite": False}) + "\n"
    )
    runner = ExperimentRunner(sut, RunnerConfig(timeout_s=5))
    run_dir = runner.run(ds, runs_root=tmp_path, run_name="demo")
    traces = load_run_traces(run_dir)
    assert [t.case_id for t in traces] == ["c1", "c2"]
    assert traces[0].final_answer and traces[1].final_answer == "I don't know."


# ---------------------------------------------------------------- engine


def _make_run(tmp_path: Path, style: str = "faithful", top_k: int = 1) -> Path:
    run_root = tmp_path
    run_root.mkdir(parents=True, exist_ok=True)
    sut = ExtractiveRAG("t", CORPUS, top_k=top_k, min_score=0.05, style=style)
    ds = run_root / "ds.jsonl"
    ds.write_text(
        json.dumps({"case_id": "c1", "input": "What is the refund window for annual plans?",
                    "expected_keywords": ["30 days"], "supporting_doc_ids": ["doc_a"]}) + "\n" +
        json.dumps({"case_id": "c2", "input": "Do you offer swahili phone support?",
                    "expect_abstention": True, "must_cite": False}) + "\n"
    )
    runner = ExperimentRunner(sut, RunnerConfig(timeout_s=5))
    run_dir = runner.run(ds, runs_root=run_root, run_name="r")
    engine = EvaluationEngine(corpus=CORPUS)
    cases = load_eval_dataset(ds)
    results = engine.evaluate_run(load_run_traces(run_dir), cases)
    from aire.evaluator.engine import write_eval_results

    write_eval_results(run_dir, results)
    return run_dir


def test_engine_aggregates_and_failures(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    results = json.loads((run_dir / "eval_results.json").read_text(encoding="utf-8"))
    agg = results["aggregates"]
    assert agg["n_cases"] == 2
    assert agg["n_success"] == 2
    assert agg["mean_citation_coverage"] == 1.0
    assert agg["failure_counts"] == {}


def test_engine_flags_missed_abstention(tmp_path: Path) -> None:
    # allow_irrelevant forces an answer even with zero retrieval score.
    run_dir = _make_run(tmp_path, abstain_bug=True)
    results = json.loads((run_dir / "eval_results.json").read_text(encoding="utf-8"))
    by_case = {r["case_id"]: r for r in results["per_case"]}
    assert by_case["c2"]["failure"] == "missed_abstention"


def test_classify_failure_priority() -> None:
    case = EvalCase(case_id="c", input="q", expected_keywords=["x"])
    trace = Trace(run_id="r", case_id="c", system_config={}, error="boom")
    assert classify_failure(trace, case, {}) == "invocation_error"


# ---------------------------------------------------------------- regression


def test_engine_flags_missed_abstention() -> None:
    """Classifier unit test: answering an unanswerable case is 'missed_abstention'."""
    case = EvalCase(case_id="c2", input="q", expect_abstention=True, must_cite=False)
    trace = Trace(run_id="r", case_id="c2", system_config={}, final_answer="Sure, we do!")
    assert classify_failure(trace, case, {}) == "missed_abstention"
    abstained_trace = Trace(run_id="r", case_id="c2", system_config={},
                            final_answer="I don't know.")
    assert classify_failure(abstained_trace, case, {}) is None


def test_compare_runs_detects_grounding_regression(tmp_path: Path) -> None:
    base = _make_run(tmp_path / "base", style="faithful", top_k=1)
    cand = _make_run(tmp_path / "cand", style="fluent", top_k=3)
    report = compare_runs(base, cand, thresholds=RegressionThresholds(default=0.0))
    assert report["summary"]["n_regressed"] >= 1
    verdicts = {m["metric"]: m["verdict"] for m in report["metrics"]}
    # The fluent style drops citations: grounding must regress...
    assert verdicts.get("groundedness") == "regressed"
    assert verdicts.get("citation_coverage") == "regressed"
    assert report["summary"]["verdict"] == "regressions detected"


def test_compare_runs_requires_eval_results(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compare_runs(tmp_path / "nope1", tmp_path / "nope2")


# ---------------------------------------------------------------- report


def test_html_report_renders(tmp_path: Path) -> None:
    from aire.report import render_comparison_report, render_run_report

    run_dir = _make_run(tmp_path / "base")
    results = json.loads((run_dir / "eval_results.json").read_text(encoding="utf-8"))
    out1 = render_run_report(run_dir, results, tmp_path / "run.html")
    assert "<html" in out1.read_text(encoding="utf-8")

    comparison = compare_runs(run_dir, run_dir)
    out2 = render_comparison_report(comparison, results, results, tmp_path / "cmp.html")
    text = out2.read_text(encoding="utf-8")
    assert "no regressions detected" in text
