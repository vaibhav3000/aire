"""Deterministic answer-level metrics.

All metrics are pure functions of (trace, case) pairs (no network, no model
calls), so they are fast, exactly reproducible, and unit-testable. Probabilistic
or model-based scoring lives in judges.py, clearly separated.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..llm.deterministic import content_keywords

# A citation is "[doc:<ids>]". The "doc" prefix is case-insensitive, spaces may
# follow the colon, and one bracket may list several ids separated by commas or
# semicolons: "[doc:a, b]" and "[doc:a; doc:b]" both cite a and b.
_CITATION_RE = re.compile(r"\[\s*doc\s*:([^\]]*)\]", re.IGNORECASE)
_CITATION_ITEM_PREFIX_RE = re.compile(r"^\s*doc\s*:\s*", re.IGNORECASE)
_DOC_ID_RE = re.compile(r"^[\w.\-]+$")
_LEADING_CITATIONS_RE = re.compile(r"^(?:\[\s*doc\s*:[^\]]*\]\s*)+", re.IGNORECASE)
# Claims are split after sentence terminators and at line breaks, so every
# sentence and every bullet item needs its own citation.
_CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*\n\s*")
_APOSTROPHES = str.maketrans(
    {
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{MODIFIER LETTER APOSTROPHE}": "'",
    }
)

# The single source of truth for what counts as an abstention. Deliberately a
# fixed phrase set, not configurable: the demo SUT emits exactly one of these,
# and the real-LLM system prompt (openai_compat.py) mandates the same sentence.
# Matching ignores case, curly apostrophes, trailing punctuation and citations.
# A third-party SUT that abstains with different wording is NOT detected;
# see tests/test_aire.py::test_third_party_abstention_phrasing_is_not_detected.
ABSTENTION_PHRASES = frozenset({"i don't know", "i do not know"})

# Default share of a claim's content keywords that must occur in a cited
# document for the claim to count as supported (see groundedness()).
DEFAULT_MIN_OVERLAP = 0.5


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace for robust text comparison."""
    return " ".join(text.lower().split()) if text else ""


def exact_match(answer: str | None, expected: str) -> float:
    """1.0 iff the normalized answer equals the normalized expected string."""
    if answer is None:
        return 0.0
    return float(normalize(answer) == normalize(expected))


def keyword_in_text(keyword: str, text: str) -> bool:
    """True when `keyword` occurs in `text` on token boundaries, ignoring case.

    The match may not start inside a word or right after a '.', and may not
    end inside a word. So 'nda' does not match 'standard', '4-hour' does not
    match '24-hour', '5%' does not match '95%' and '0.1%' does not match '10.1%'.
    """
    pattern = r"(?<![\w.])" + re.escape(keyword.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def keyword_recall(answer: str | None, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer (see keyword_in_text).

    Returns 1.0 when there are no expectations (vacuous success is deliberate:
    the engine only reports metrics the case actually specifies).
    """
    if not expected_keywords:
        return 1.0
    if not answer:
        return 0.0
    hits = sum(1 for k in expected_keywords if keyword_in_text(k, answer))
    return hits / len(expected_keywords)


def abstained(answer: str | None) -> bool:
    """True when the answer is one of ABSTENTION_PHRASES.

    A missing or empty answer is not an abstention: it is a failed invocation
    and the engine scores it as one.

    Coupling note: recognition is exact-phrase against ABSTENTION_PHRASES after
    normalising case, whitespace, curly apostrophes, trailing punctuation and
    citations. Other refusal wordings are treated as answers.
    """
    if not answer:
        return False
    text = normalize(_CITATION_RE.sub(" ", answer.translate(_APOSTROPHES)))
    return text.rstrip(" .!") in ABSTENTION_PHRASES


def citation_ids(answer: str | None) -> list[str]:
    """Extract cited doc ids in order of appearance (see _CITATION_RE)."""
    if not answer:
        return []
    ids: list[str] = []
    for match in _CITATION_RE.finditer(answer):
        for item in re.split(r"[,;]", match.group(1)):
            doc_id = _CITATION_ITEM_PREFIX_RE.sub("", item).strip()
            if _DOC_ID_RE.match(doc_id):
                ids.append(doc_id)
    return ids


def gold_citation_recall(answer: str | None, supporting_doc_ids: list[str]) -> float:
    """Fraction of the gold supporting docs that the answer cites anywhere.

    This is recall over gold docs only: citing extra documents does not lower
    it, which is why citation_precision exists.
    """
    if not supporting_doc_ids:
        return 1.0
    cited = set(citation_ids(answer))
    return sum(1 for d in supporting_doc_ids if d in cited) / len(supporting_doc_ids)


def citation_precision(answer: str | None, allowed_doc_ids: Iterable[str]) -> float | None:
    """Share of distinct cited docs that are allowed (gold or retrieved).

    None when the answer cites nothing: precision is not measurable there, and
    the missing citations already show up in gold_citation_recall.
    """
    cited = set(citation_ids(answer))
    if not cited:
        return None
    return len(cited & set(allowed_doc_ids)) / len(cited)


def _claims(answer: str) -> list[str]:
    """Split an answer into claims, each carrying the citations that follow it.

    Citations directly after a sentence terminator belong to the sentence
    before them ("S1. [doc:x] S2." cites S1), following the convention that a
    citation follows the claim it supports.
    """
    claims: list[str] = []
    for raw in _CLAIM_SPLIT_RE.split(answer):
        piece = raw.strip()
        if not piece:
            continue
        lead = _LEADING_CITATIONS_RE.match(piece)
        if lead and claims:
            claims[-1] = f"{claims[-1]} {lead.group(0).strip()}"
            piece = piece[lead.end() :].strip()
            if not piece:
                continue
        claims.append(piece)
    return claims


def groundedness(
    answer: str | None,
    corpus: Mapping[str, str],
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> float | None:
    """Deterministic citation-groundedness proxy in [0, 1]; None when not measurable.

    The answer is split into claims (sentences and lines). A claim is supported
    when at least `min_overlap` of its content keywords occur in a document it
    cites. Score = supported claims / claims with content keywords. Uncited
    claims count as unsupported, and so do citations of documents missing from
    `corpus`: the engine passes only the documents the system retrieved, so an
    answer cannot be grounded in context it never had. None when the answer is
    empty, abstained, or makes no claim.

    This is a LEXICAL proxy, not semantic entailment: it is cheap, auditable
    and deterministic, but it can be fooled by paraphrases with overlapping
    vocabulary. The design deliberately exposes that limitation; see
    docs/design_faq.md.
    """
    if not answer or abstained(answer):
        return None
    supported = 0
    checked = 0
    for claim in _claims(answer):
        claim_keywords = content_keywords(_CITATION_RE.sub(" ", claim))
        if not claim_keywords:
            continue  # no factual content, e.g. a bare citation
        checked += 1
        for doc_id in citation_ids(claim):
            doc_text = corpus.get(doc_id)
            if doc_text is None:
                continue
            overlap = len(claim_keywords & content_keywords(doc_text)) / len(claim_keywords)
            if overlap >= min_overlap:
                supported += 1
                break
    if checked == 0:
        return None
    return supported / checked


def cost_estimate(usage_total_tokens: int, price_per_1k_tokens: float) -> float:
    """USD cost estimate from token usage and a price table entry."""
    return round(usage_total_tokens / 1000.0 * price_per_1k_tokens, 6)


def metric_spec() -> dict[str, dict[str, Any]]:
    """Declarative metric registry used by the engine, the regression gate and reports.

    direction: "higher" | "lower", needed by the regression engine.
    bounded:   True for scores in [0, 1], which the gate compares with an
               absolute tolerance; False for latency, tokens and cost, which
               it compares relative to the baseline.
    Retrieval metrics are reported per cut-off as retrieval_recall_at_<k> and
    retrieval_mrr_at_<k>; metric_info() resolves those names.
    """
    return {
        "keyword_recall": {"direction": "higher", "kind": "quality", "bounded": True},
        "exact_match": {"direction": "higher", "kind": "quality", "bounded": True},
        "gold_citation_recall": {"direction": "higher", "kind": "quality", "bounded": True},
        "citation_precision": {"direction": "higher", "kind": "quality", "bounded": True},
        "groundedness": {"direction": "higher", "kind": "quality", "bounded": True},
        "abstention_correct": {"direction": "higher", "kind": "quality", "bounded": True},
        "abstention_accuracy": {"direction": "higher", "kind": "quality", "bounded": True},
        "false_refusal": {"direction": "lower", "kind": "quality", "bounded": True},
        "retrieval_recall": {"direction": "higher", "kind": "retrieval", "bounded": True},
        "retrieval_mrr": {"direction": "higher", "kind": "retrieval", "bounded": True},
        "tool_selection_correct": {"direction": "higher", "kind": "tool", "bounded": True},
        "tool_args_valid": {"direction": "higher", "kind": "tool", "bounded": True},
        "tool_args_match": {"direction": "higher", "kind": "tool", "bounded": True},
        "tool_selection_accuracy": {"direction": "higher", "kind": "tool", "bounded": True},
        "tool_call_precision": {"direction": "higher", "kind": "tool", "bounded": True},
        "step_efficiency": {"direction": "higher", "kind": "tool", "bounded": True},
        "tool_args_schema_valid": {"direction": "higher", "kind": "tool", "bounded": True},
        "n_steps": {"direction": "lower", "kind": "tool", "bounded": False},
        "judge_correct": {"direction": "higher", "kind": "judge", "bounded": True},
        "judge_supported": {"direction": "higher", "kind": "judge", "bounded": True},
        "latency_ms": {"direction": "lower", "kind": "cost", "bounded": False},
        "cost_usd": {"direction": "lower", "kind": "cost", "bounded": False},
        "total_tokens": {"direction": "lower", "kind": "cost", "bounded": False},
        "errors": {"direction": "lower", "kind": "reliability", "bounded": True},
        "success_rate": {"direction": "higher", "kind": "reliability", "bounded": True},
    }


_AT_K_RE = re.compile(r"^(?P<base>.+)_at_\d+$")


def metric_info(metric: str) -> dict[str, Any] | None:
    """Registry entry for a metric name, resolving '<base>_at_<k>' names; None if unknown."""
    spec = metric_spec()
    if metric in spec:
        return spec[metric]
    match = _AT_K_RE.match(metric)
    if match:
        return spec.get(match.group("base"))
    return None
