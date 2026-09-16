"""Deterministic answer-level metrics.

All metrics are pure functions of (trace, case) pairs — no network, no model
calls — so they are fast, exactly reproducible, and unit-testable. Probabilistic
or model-based scoring lives in judges.py, clearly separated.
"""

from __future__ import annotations

import re
from typing import Any

from ..llm.deterministic import content_keywords

_CITATION_RE = re.compile(r"\[doc:([\w.\-]+)\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace for robust text comparison."""
    return " ".join(text.lower().split()) if text else ""


def exact_match(answer: str | None, expected: str) -> float:
    """1.0 iff the normalized answer equals the normalized expected string."""
    if answer is None:
        return 0.0
    return float(normalize(answer) == normalize(expected))


def keyword_recall(answer: str | None, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present (case-insensitive) in the answer.

    Returns 1.0 when there are no expectations (vacuous success is deliberate:
    the engine only reports metrics the case actually specifies).
    """
    if not expected_keywords:
        return 1.0
    if not answer:
        return 0.0
    ans = answer.lower()
    hits = sum(1 for k in expected_keywords if k.lower() in ans)
    return hits / len(expected_keywords)


def abstained(answer: str | None) -> bool:
    """True when the answer is the configured abstention phrase or empty."""
    if not answer:
        return True
    return normalize(answer) in {"i don't know", "i do not know", "i don't know."}


def citation_ids(answer: str | None) -> list[str]:
    """Extract [doc:<id>] citations from an answer."""
    if not answer:
        return []
    return _CITATION_RE.findall(answer)


def citation_coverage(answer: str | None, supporting_doc_ids: list[str]) -> float:
    """Fraction of supporting docs that are cited in the answer."""
    if not supporting_doc_ids:
        return 1.0
    cited = set(citation_ids(answer))
    return sum(1 for d in supporting_doc_ids if d in cited) / len(supporting_doc_ids)


def json_validity(answer: str | None) -> float | None:
    """1.0/0.0 if the answer is expected to be JSON; None when not applicable.

    Applicability is decided by the engine (case expectations), not here.
    """
    if answer is None or not answer.strip():
        return 0.0
    import json

    try:
        json.loads(answer)
        return 1.0
    except json.JSONDecodeError:
        return 0.0


def groundedness(answer: str | None, corpus: dict[str, str]) -> float | None:
    """Deterministic citation-groundedness proxy in [0, 1]; None when not measurable.

    For each cited sentence of the answer, check whether its content keywords
    appear in the cited document. Score = supported cited sentences / total
    cited sentences. Answers with no citations score 0.0 when they make factual
    claims (any sentence with content keywords) and None only when the answer
    is empty/abstained.

    This is a LEXICAL proxy, not semantic entailment: it is cheap, auditable
    and deterministic, but it can be fooled by paraphrases with overlapping
    vocabulary. The design deliberately exposes that limitation; see
    docs/interview_guide.md.
    """
    if not answer or abstained(answer):
        return None
    raw_sentences = [s.strip() for s in _SENT_SPLIT_RE.split(answer) if s.strip()]
    if not raw_sentences:
        return None
    # A segment that consists ONLY of citation tokens is an attachment to the
    # previous claim (answers commonly append "[doc:x]" after the final period).
    segments: list[str] = []
    for sentence in raw_sentences:
        if (
            segments
            and citation_ids(sentence)
            and not content_keywords(_CITATION_RE.sub("", sentence))
        ):
            segments[-1] = segments[-1] + " " + sentence
        else:
            segments.append(sentence)

    supported = 0
    checked = 0
    for segment in segments:
        cited = citation_ids(segment)
        if not cited:
            # Uncited factual sentence: count it as unsupported claim.
            if content_keywords(_CITATION_RE.sub("", segment)):
                checked += 1
            continue
        checked += 1
        sent_keywords = content_keywords(_CITATION_RE.sub("", segment))
        for doc_id in cited:
            doc_text = corpus.get(doc_id)
            if doc_text is None:
                continue  # citation to a nonexistent doc is checked at coverage level
            doc_keywords = content_keywords(doc_text)
            if sent_keywords and len(sent_keywords & doc_keywords) / len(sent_keywords) >= 0.5:
                supported += 1
                break
    if checked == 0:
        return None
    return supported / checked


def cost_estimate(usage_total_tokens: int, price_per_1k_tokens: float) -> float:
    """USD cost estimate from token usage and a price table entry."""
    return round(usage_total_tokens / 1000.0 * price_per_1k_tokens, 6)


def metric_spec() -> dict[str, dict[str, Any]]:
    """Declarative metric registry used by the engine and the report.

    direction: "higher" | "lower" — needed by the regression engine.
    """
    return {
        "keyword_recall": {"direction": "higher", "kind": "quality"},
        "exact_match": {"direction": "higher", "kind": "quality"},
        "citation_coverage": {"direction": "higher", "kind": "quality"},
        "groundedness": {"direction": "higher", "kind": "quality"},
        "abstention_correct": {"direction": "higher", "kind": "quality"},
        "retrieval_recall": {"direction": "higher", "kind": "retrieval"},
        "retrieval_mrr": {"direction": "higher", "kind": "retrieval"},
        "tool_selection_correct": {"direction": "higher", "kind": "tool"},
        "tool_args_valid": {"direction": "higher", "kind": "tool"},
        "latency_ms": {"direction": "lower", "kind": "cost"},
        "cost_usd": {"direction": "lower", "kind": "cost"},
        "total_tokens": {"direction": "lower", "kind": "cost"},
        "errors": {"direction": "lower", "kind": "reliability"},
    }
