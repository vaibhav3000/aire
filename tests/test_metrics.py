"""Deterministic answer-level metrics: groundedness, citations, keywords, abstention."""

import pytest

from aire.evaluator import metrics as M
from aire.llm.deterministic import ExtractiveRAG

DOC_A = "The refund window is 30 days for annual plans. Credits are issued to the account."
CORPUS = {"doc_a": DOC_A, "doc_b": "API rate limits are 60 requests per minute."}


# ---------------------------------------------------------------- groundedness


def test_verbatim_extractive_answer_is_fully_grounded() -> None:
    """ExtractiveRAG quotes two sentences as 'S1. [doc:x] S2. [doc:x]'. Both are
    verbatim from the cited document, so the answer is fully grounded."""
    sut = ExtractiveRAG("t", CORPUS, top_k=1, min_score=0.05)
    out = sut.invoke("How are credits issued for the refund window?")
    assert out.answer is not None
    assert out.answer.count("[doc:doc_a]") == 2
    assert M.groundedness(out.answer, CORPUS) == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        "The refund window is 30 days for annual plans. [doc:doc_a] "
        "Credits are issued to the account. [doc:doc_a]",
        "The refund window is 30 days for annual plans [doc:doc_a]. "
        "Credits are issued to the account [doc:doc_a].",
    ],
)
def test_citation_placement_around_the_period_does_not_change_groundedness(answer: str) -> None:
    assert M.groundedness(answer, CORPUS) == 1.0


def test_trailing_citation_binds_only_to_the_sentence_before_it() -> None:
    answer = (
        "The refund window is 30 days for annual plans. Credits are issued to the account. "
        "[doc:doc_a]"
    )
    assert M.groundedness(answer, CORPUS) == 0.5


def test_one_trailing_citation_does_not_cover_every_bullet() -> None:
    answer = (
        "- The refund window is 30 days for annual plans\n"
        "- Credits are issued to the account [doc:doc_a]"
    )
    assert M.groundedness(answer, CORPUS) == 0.5


def test_groundedness_overlap_threshold_is_configurable() -> None:
    # 6 of the sentence's 9 content keywords occur in doc_a.
    answer = (
        "The refund window is 30 days for annual plans and includes lifetime upgrades [doc:doc_a]."
    )
    assert M.groundedness(answer, CORPUS) == 1.0
    assert M.groundedness(answer, CORPUS, min_overlap=0.9) == 0.0


# ---------------------------------------------------------------- keywords


@pytest.mark.parametrize(
    ("answer", "keyword"),
    [
        ("The standard plan", "nda"),
        ("Logs are purged after the 24-hour window", "4-hour"),
        ("Credits of 95% apply", "5%"),
        ("Downtime of 10.1% is rare", "0.1%"),
        ("Records are kept for 130 days", "30 days"),
    ],
)
def test_keyword_inside_a_longer_token_does_not_count(answer: str, keyword: str) -> None:
    assert M.keyword_recall(answer, [keyword]) == 0.0


@pytest.mark.parametrize(
    ("answer", "keyword"),
    [
        ("The report is available under NDA.", "nda"),
        ("Priority support has a 4-hour target", "4-hour"),
        ("A credit of 5% applies", "5%"),
        ("Every 0.1% of downtime counts", "0.1%"),
        ("Logs are kept for 30 days, then removed", "30 days"),
        ("See status.northwind.cloud for updates", "status.northwind.cloud"),
    ],
)
def test_keyword_matches_on_token_boundaries(answer: str, keyword: str) -> None:
    assert M.keyword_recall(answer, [keyword]) == 1.0


# ---------------------------------------------------------------- citations


def test_citation_parsing_accepts_common_variants() -> None:
    answer = "A [doc:a,b]. B [doc: c]. C [DOC:d]. D [doc:e; doc:f]. E [doc:doc_g]."
    assert M.citation_ids(answer) == ["a", "b", "c", "d", "e", "f", "doc_g"]


def test_citation_precision_penalises_citation_spam() -> None:
    spam = "Refunds take 30 days [doc:doc_a][doc:doc_b][doc:doc_c][doc:doc_d]."
    assert M.citation_precision(spam, ["doc_a"]) == 0.25
    assert M.citation_precision("Refunds take 30 days [doc:doc_a].", ["doc_a"]) == 1.0


def test_citation_precision_is_not_measurable_without_citations() -> None:
    assert M.citation_precision("Refunds take 30 days.", ["doc_a"]) is None


# ---------------------------------------------------------------- abstention


@pytest.mark.parametrize(
    "answer",
    [
        "I don\N{RIGHT SINGLE QUOTATION MARK}t know.",
        "I don't know. [doc:doc_a]",
        "I do not know.",
        "  i don't know  ",
    ],
)
def test_abstention_variants_are_recognised(answer: str) -> None:
    assert M.abstained(answer) is True


@pytest.mark.parametrize("answer", [None, ""])
def test_a_missing_answer_is_not_an_abstention(answer: str | None) -> None:
    assert M.abstained(answer) is False
