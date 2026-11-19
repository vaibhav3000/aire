"""The shared RAG prompt keeps the format the Step 1 responder sent."""

from aire.systems.prompts import rag_user_prompt


def test_documents_come_first_then_the_question() -> None:
    prompt = rag_user_prompt("When?", [("doc_a", "Text A."), ("doc_b", "Text B.")])
    assert (
        prompt
        == "Context documents:\n[doc:doc_a]\nText A.\n\n[doc:doc_b]\nText B.\n\nQuestion: When?"
    )


def test_no_documents_is_said_explicitly() -> None:
    assert (
        rag_user_prompt("When?", [])
        == "Context documents:\n(no documents found)\n\nQuestion: When?"
    )
