"""Rubric and pairwise judges, the lexical baseline judge, calibration and hybrid scoring.

Judge models are ScriptedChatModels: every verdict is decided by the test.
"""

import asyncio
import json
import re

import pytest

from aire.evaluator.calibration import agreement
from aire.evaluator.engine import EvaluationEngine
from aire.evaluator.judges import (
    LexicalRubricJudge,
    PairwiseJudge,
    RubricJudge,
    apply_rubric_verdicts,
    pairwise_summary,
)
from aire.evaluator.metrics import metric_info
from aire.providers.client import ModelClient
from aire.providers.fake import ScriptedChatModel
from aire.providers.types import ChatRequest, ChatResponse, TokenUsage
from aire.runner.dataset import EvalCase
from aire.schema import Trace

DOCS = [("doc_refunds", "Refunds are allowed within 30 days of a charge.")]


def _reply(payload: dict) -> ChatResponse:
    return ChatResponse(
        text=json.dumps(payload),
        usage=TokenUsage(input_tokens=100, output_tokens=20),
        response_model="judge-model",
        finish_reason="stop",
        parsed=payload,
    )


def _client(model: ScriptedChatModel) -> ModelClient:
    return ModelClient(model)


def test_rubric_judge_asks_for_structured_criteria_and_records_its_identity() -> None:
    model = ScriptedChatModel(
        [_reply({"correct": True, "supported": False, "rationale": "Right fact, no citation."})]
    )
    judge = RubricJudge(_client(model), model="judge-model")
    verdict = asyncio.run(
        judge.judge("How long for refunds?", "30 days.", reference=DOCS, retrieved=DOCS)
    )
    assert (verdict.correct, verdict.supported) == (True, False)
    request: ChatRequest = model.requests[0]
    assert request.json_schema is not None
    assert set(request.json_schema["required"]) == {"correct", "supported", "rationale"}
    assert request.temperature == 0.0
    assert "doc_refunds" in request.messages[-1].content
    identity = judge.identity()
    assert identity["model"] == "judge-model"
    assert identity["version"] == RubricJudge.VERSION
    assert len(identity["prompt_sha256"]) == 12


def _prefers_cited(request: ChatRequest) -> ChatResponse:
    """A fair fake judge: picks whichever answer carries a citation, wherever it is."""
    prompt = request.messages[-1].content
    a = re.search(r"Answer A:\n(.*?)\n\nAnswer B:", prompt, re.S)
    assert a is not None
    winner = "A" if "[doc:" in a.group(1) else "B"
    return _reply({"winner": winner, "rationale": "cites its source"})


def test_pairwise_judge_asks_both_orders_and_keeps_a_consistent_winner() -> None:
    model = ScriptedChatModel(default=_prefers_cited)
    judge = PairwiseJudge(_client(model), model="judge-model")
    verdict = asyncio.run(
        judge.compare("Refunds?", "30 days [doc:doc_refunds].", "30 days.", reference=DOCS)
    )
    assert verdict.winner == "a"
    assert verdict.consistent
    assert len(model.requests) == 2
    first, second = (r.messages[-1].content for r in model.requests)
    assert first.index("30 days [doc:") < first.index("Answer B:")
    assert second.index("30 days [doc:") > second.index("Answer B:")


def test_a_judge_that_always_picks_the_first_position_only_produces_ties() -> None:
    model = ScriptedChatModel(default=lambda r: _reply({"winner": "A", "rationale": "first"}))
    judge = PairwiseJudge(_client(model), model="judge-model")
    verdicts = [
        asyncio.run(judge.compare("q", f"answer {i}", f"other {i}", reference=DOCS))
        for i in range(3)
    ]
    assert {v.winner for v in verdicts} == {"tie"}
    summary = pairwise_summary(verdicts)
    assert summary["position_consistency"] == 0.0
    assert summary["ties"] == 3


def test_pairwise_summary_counts_wins() -> None:
    model = ScriptedChatModel(default=_prefers_cited)
    judge = PairwiseJudge(_client(model), model="judge-model")
    verdicts = [
        asyncio.run(judge.compare("q", "x [doc:a].", "y.", reference=DOCS)),
        asyncio.run(judge.compare("q", "x.", "y [doc:a].", reference=DOCS)),
        asyncio.run(judge.compare("q", "x.", "y [doc:a].", reference=DOCS)),
    ]
    summary = pairwise_summary(verdicts)
    assert (summary["wins_a"], summary["wins_b"], summary["ties"]) == (1, 2, 0)
    assert summary["win_rate_b"] == round(2 / 3, 4)  # summaries are rounded to 4 places
    assert summary["position_consistency"] == 1.0


def test_lexical_judge_reads_the_deterministic_metrics() -> None:
    judge = LexicalRubricJudge()
    answered = {"keyword_recall": 0.67, "groundedness": 1.0, "abstention_correct": None}
    uncited = {"keyword_recall": 1.0, "groundedness": 0.0, "abstention_correct": None}
    abstained_right = {"abstention_correct": 1.0}
    assert judge.from_row(answered, expect_abstention=False) == (True, True)
    assert judge.from_row(uncited, expect_abstention=False) == (True, False)
    assert judge.from_row(abstained_right, expect_abstention=True) == (True, True)


def test_agreement_reports_kappa_and_the_confusion_table() -> None:
    report = agreement([True, True, False, False], [True, False, False, False])
    assert report["n"] == 4
    assert report["agreement"] == 0.75
    assert report["confusion"] == {"yes/yes": 1, "yes/no": 1, "no/yes": 0, "no/no": 2}
    assert report["kappa"] == pytest.approx(0.5)


def test_judge_verdicts_join_the_deterministic_metrics_and_are_gated() -> None:
    case = EvalCase(case_id="c1", input="q", expected_keywords=["30 days"], must_cite=False)
    trace = Trace(run_id="r", case_id="c1", system_config={}, final_answer="30 days.")
    engine = EvaluationEngine()
    results = engine.evaluate_run([trace], [case])
    keyword_before = results["per_case"][0]["keyword_recall"]
    judged = apply_rubric_verdicts(
        results,
        {"c1": (True, False)},
        identity={"judge": "rubric_llm", "model": "m"},
        engine=engine,
    )
    row = judged["per_case"][0]
    assert row["keyword_recall"] == keyword_before  # deterministic metrics are kept
    assert (row["judge_correct"], row["judge_supported"]) == (1.0, 0.0)
    assert judged["aggregates"]["mean_judge_correct"] == 1.0
    assert judged["judges"] == [{"judge": "rubric_llm", "model": "m"}]
    for name in ("judge_correct", "judge_supported"):
        info = metric_info(name)
        assert info is not None and info["direction"] == "higher"


def test_lexical_judge_scores_an_answer_directly_against_retrieved_documents() -> None:
    judge = LexicalRubricJudge()
    corpus = {"doc_refunds": "Refunds are allowed within 30 days of a charge."}
    cited = "Refunds are allowed within 30 days of a charge. [doc:doc_refunds]"
    uncited = "Refunds are allowed within 30 days of a charge."
    assert judge.judge_answer(cited, ["30 days"], False, corpus) == (True, True)
    assert judge.judge_answer(uncited, ["30 days"], False, corpus) == (True, False)
    # An unanswerable question: quoting a cited document is supported but not correct.
    assert judge.judge_answer(cited, [], True, corpus) == (False, True)
    assert judge.judge_answer("I don't know.", [], True, corpus) == (True, True)
    assert judge.judge_answer("I don't know.", ["30 days"], False, corpus) == (False, True)
