"""The deterministic demo system (ExtractiveRAG): what its answers look like."""

from aire.evaluator import metrics as M
from aire.llm.deterministic import ExtractiveRAG

CORPUS = {
    "doc_enc": (
        "# Encryption at Rest\n\n"
        "Northwind Cloud encrypts stored data with AES-256 keys. "
        "Keys for stored data rotate every 90 days."
    ),
    "doc_billing": "# Billing\n\nPayments are processed monthly.",
}
QUESTION = "How is stored data encrypted and how often do keys rotate?"


def test_answers_do_not_quote_markdown_headings() -> None:
    sut = ExtractiveRAG("t", CORPUS, top_k=1)
    answer = sut.invoke(QUESTION).answer
    assert answer is not None
    assert "#" not in answer and "Encryption at Rest" not in answer


def test_fluent_style_keeps_the_original_capitalisation() -> None:
    sut = ExtractiveRAG("t", CORPUS, top_k=1, style="fluent")
    answer = sut.invoke(QUESTION).answer
    assert answer is not None
    assert "Northwind Cloud" in answer and "AES-256" in answer
    assert "[doc:" not in answer


def test_fluent_style_abstains_when_no_sentence_matches() -> None:
    # Only the heading mentions billing, so no quotable sentence overlaps.
    sut = ExtractiveRAG("t", CORPUS, top_k=1, style="fluent")
    answer = sut.invoke("Billing question?").answer
    assert M.abstained(answer)
