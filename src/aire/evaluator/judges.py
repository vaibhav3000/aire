"""Judges: scoring functions that go beyond deterministic string checks.

Design position: deterministic metrics are never replaced, only joined.
AIRE ships a deterministic groundedness judge, a deterministic rubric
baseline (LexicalRubricJudge) and two model-based judges: RubricJudge
(correct and supported, as structured output) and PairwiseJudge (asked in
both orders, reporting position consistency). We do NOT oversell
LLM-as-judge: every set of verdicts records the judge's identity (model,
version, prompt hash, temperature), docs/design_faq.md discusses position
bias, verbosity bias, self-preference and cost, and a judge is calibrated
against labels (evaluator.calibration) before its numbers are trusted.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..llm.base import Judge
from ..providers.client import ModelClient
from ..providers.types import ChatRequest, Message
from ..schema import Trace
from .engine import EvaluationEngine
from .metrics import abstained, keyword_recall
from .metrics import groundedness as lexical_groundedness


class GroundednessJudge(Judge):
    """Deterministic citation-groundedness judge (lexical overlap proxy).

    See evaluator.metrics.groundedness for the exact definition and its limits.
    """

    name = "groundedness_lexical"

    def __init__(self, corpus: dict[str, str]) -> None:
        self.corpus = corpus

    def score(self, trace: Trace, context: dict[str, Any]) -> dict[str, Any]:
        # Only documents the system actually retrieved can ground its answer.
        retrieved = {d: self.corpus[d] for d in trace.retrieved_doc_ids if d in self.corpus}
        value = lexical_groundedness(trace.final_answer, retrieved)
        if value is None:
            detail = "not measurable (empty or abstained answer)"
        else:
            detail = "lexical citation-overlap proxy; deterministic"
        return {"score": value, "detail": detail}


class LLMJudge(Judge):
    """Pluggable model-based judge. NOT used by committed demo results.

    The scorer receives the full trace dict and returns a score in [0,1].
    It is the integrator's responsibility to evaluate the judge itself
    (agreement with human labels) before trusting it; AIRE records judge
    metadata to make that audit possible.
    """

    name = "llm_judge"

    def __init__(self, scorer: Any, description: str, version: str = "unversioned") -> None:
        self.scorer = scorer  # callable(trace_dict, case_dict) -> float in [0,1]
        self.description = description
        self.version = version

    def score(self, trace: Trace, context: dict[str, Any]) -> dict[str, Any]:
        trace_dict = trace.to_dict()
        value = float(self.scorer(trace_dict, context))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"LLMJudge returned out-of-range score {value}")
        return {
            "score": value,
            "detail": {
                "judge": self.description,
                "version": self.version,
                "warning": "model-based judge; validate against human labels before trusting",
            },
        }


# --- model-based judges (Step 2) ----------------------------------------------

Documents = Sequence[tuple[str, str]]

_RUBRIC_SYSTEM = (
    "You grade answers from a customer-support assistant. You are given a "
    "question, reference documents that contain the facts needed to answer it "
    "(none means the question cannot be answered from the documents), the "
    "documents the assistant retrieved, and the assistant's answer. Decide two "
    "things independently. correct: the answer gives the information the "
    "question asks for and agrees with the reference documents; when there are "
    "no reference documents, only an answer that declines (for example 'I don't "
    "know.') is correct. supported: every factual claim in the answer appears in "
    "a retrieved document that the answer cites in the form [doc:<id>]; an answer "
    "that only declines is supported. Ignore style and length. Reply with JSON only."
)
_RUBRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "supported": {"type": "boolean"},
        "rationale": {"type": "string", "maxLength": 600},
    },
    "required": ["correct", "supported", "rationale"],
    "additionalProperties": False,
}
_PAIRWISE_SYSTEM = (
    "You compare two answers from a customer-support assistant to the same "
    "question, using the reference documents. Prefer the answer that is correct "
    "and complete, then the one whose claims are cited to the documents. Do not "
    "prefer an answer for its position or its length. If neither is better, say "
    "tie. Reply with JSON only."
)
_PAIRWISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "winner": {"enum": ["A", "B", "tie"]},
        "rationale": {"type": "string", "maxLength": 600},
    },
    "required": ["winner", "rationale"],
    "additionalProperties": False,
}


def _documents_block(title: str, documents: Documents) -> str:
    if not documents:
        return f"{title}: none"
    body = "\n\n".join(f"[doc:{doc_id}]\n{text}" for doc_id, text in documents)
    return f"{title}:\n{body}"


def _prompt_hash(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class RubricVerdict:
    correct: bool
    supported: bool
    rationale: str


class RubricJudge:
    """Asks a model the two rubric criteria for one answer, as structured output.

    The judge's identity (model, rubric version, prompt hash, temperature) is
    recorded with every set of verdicts, so a changed prompt is a different
    judge and its calibration must be redone.
    """

    VERSION = "rubric-v1"

    def __init__(self, client: ModelClient, model: str, *, temperature: float | None = 0.0) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def identity(self) -> dict[str, Any]:
        return {
            "judge": "rubric_llm",
            "provider": self.client.provider,
            "model": self.model,
            "version": self.VERSION,
            "prompt_sha256": _prompt_hash(
                _RUBRIC_SYSTEM, json.dumps(_RUBRIC_SCHEMA, sort_keys=True)
            ),
            "temperature": self.temperature,
        }

    async def judge(
        self, question: str, answer: str, *, reference: Documents, retrieved: Documents
    ) -> RubricVerdict:
        user = "\n\n".join(
            [
                f"Question: {question}",
                _documents_block("Reference documents", reference),
                _documents_block("Retrieved documents", retrieved),
                f"Answer to grade:\n{answer}",
            ]
        )
        request = ChatRequest(
            model=self.model,
            messages=(Message("system", _RUBRIC_SYSTEM), Message("user", user)),
            max_output_tokens=400,
            temperature=self.temperature,
            json_schema=_RUBRIC_SCHEMA,
        )
        parsed = (await self.client.complete(request)).response.parsed
        return RubricVerdict(
            correct=bool(parsed["correct"]),
            supported=bool(parsed["supported"]),
            rationale=str(parsed["rationale"]),
        )


@dataclass(frozen=True)
class PairwiseVerdict:
    winner: str  # "a", "b" or "tie"
    consistent: bool  # both orders named the same winner
    a_first: str  # raw reply with answer a shown as A
    b_first: str  # raw reply with answer b shown as A


class PairwiseJudge:
    """Compares two answers twice, swapping their positions.

    A winner counts only when both orders agree; otherwise the comparison is a
    tie and counts against position consistency, which measures how much the
    judge's choice depends on where an answer is shown.
    """

    VERSION = "pairwise-v1"

    def __init__(self, client: ModelClient, model: str, *, temperature: float | None = 0.0) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def identity(self) -> dict[str, Any]:
        return {
            "judge": "pairwise_llm",
            "provider": self.client.provider,
            "model": self.model,
            "version": self.VERSION,
            "prompt_sha256": _prompt_hash(
                _PAIRWISE_SYSTEM, json.dumps(_PAIRWISE_SCHEMA, sort_keys=True)
            ),
            "temperature": self.temperature,
        }

    async def _ask(self, question: str, first: str, second: str, reference: Documents) -> str:
        user = "\n\n".join(
            [
                f"Question: {question}",
                _documents_block("Reference documents", reference),
                f"Answer A:\n{first}",
                f"Answer B:\n{second}",
            ]
        )
        request = ChatRequest(
            model=self.model,
            messages=(Message("system", _PAIRWISE_SYSTEM), Message("user", user)),
            max_output_tokens=300,
            temperature=self.temperature,
            json_schema=_PAIRWISE_SCHEMA,
        )
        return str((await self.client.complete(request)).response.parsed["winner"])

    async def compare(
        self, question: str, answer_a: str, answer_b: str, *, reference: Documents
    ) -> PairwiseVerdict:
        a_first = await self._ask(question, answer_a, answer_b, reference)
        b_first = await self._ask(question, answer_b, answer_a, reference)
        straight = {"A": "a", "B": "b", "tie": "tie"}[a_first]
        swapped_back = {"A": "b", "B": "a", "tie": "tie"}[b_first]
        consistent = straight == swapped_back
        return PairwiseVerdict(
            winner=straight if consistent else "tie",
            consistent=consistent,
            a_first=a_first,
            b_first=b_first,
        )


def pairwise_summary(verdicts: Sequence[PairwiseVerdict]) -> dict[str, Any]:
    n = len(verdicts)
    wins_a = sum(1 for v in verdicts if v.winner == "a")
    wins_b = sum(1 for v in verdicts if v.winner == "b")
    return {
        "n": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": n - wins_a - wins_b,
        "win_rate_b": round(wins_b / n, 4) if n else None,
        "position_consistency": round(sum(v.consistent for v in verdicts) / n, 4) if n else None,
    }


class LexicalRubricJudge:
    """The deterministic baseline with the rubric's two criteria.

    correct: keyword recall of at least 0.5 on an answerable case that was not
    declined, or a correct abstention on an unanswerable case.
    supported: groundedness of at least 0.5, or an answer that only declines.
    Calibrating it against labels shows what the lexical proxies miss.
    """

    def identity(self) -> dict[str, Any]:
        return {"judge": "rubric_lexical", "version": "lexical-v1", "threshold": 0.5}

    def judge_answer(
        self,
        answer: str | None,
        expected_keywords: Sequence[str],
        expect_abstention: bool,
        retrieved: Mapping[str, str],
    ) -> tuple[bool, bool]:
        """(correct, supported) for one answer, from the deterministic metrics.

        `retrieved` maps the ids of the documents the system retrieved to their
        text; groundedness is measured against those only.
        """
        declined = abstained(answer)
        if expect_abstention:
            correct = declined
        else:
            correct = not declined and keyword_recall(answer, list(expected_keywords)) >= 0.5
        grounded = lexical_groundedness(answer, retrieved)
        supported = declined or (grounded is not None and grounded >= 0.5)
        return correct, supported

    def from_row(self, row: dict[str, Any], *, expect_abstention: bool) -> tuple[bool, bool]:
        if expect_abstention:
            declined = row.get("abstention_correct") == 1.0
            return declined, declined or (row.get("groundedness") or 0.0) >= 0.5
        if row.get("false_refusal") == 1.0:
            return False, True
        correct = (row.get("keyword_recall") or 0.0) >= 0.5
        supported = (row.get("groundedness") or 0.0) >= 0.5
        return correct, supported


def apply_rubric_verdicts(
    results: dict[str, Any],
    verdicts: Mapping[str, tuple[bool, bool] | None],
    *,
    identity: dict[str, Any],
    engine: EvaluationEngine,
) -> dict[str, Any]:
    """Join rubric verdicts to scored results as judge_correct and judge_supported.

    Deterministic metrics stay untouched; a case the judge could not grade
    (verdict None) gets no judge metrics and is counted in judge_errors.
    """
    judged = copy.deepcopy(results)
    errors = 0
    for row in judged["per_case"]:
        if row["case_id"] not in verdicts:
            continue
        verdict = verdicts[row["case_id"]]
        if verdict is None:
            errors += 1
            continue
        row["judge_correct"] = float(verdict[0])
        row["judge_supported"] = float(verdict[1])
    judged["aggregates"] = engine.aggregate(judged["per_case"])
    judged["aggregates"]["judge_errors"] = errors
    judged.setdefault("judges", []).append(identity)
    return judged
