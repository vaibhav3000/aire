"""The RAG prompt shared by every LLM-backed system and the fake server.

Citations in AIRE's [doc:<id>] form and the exact abstention sentence are
required, so the deterministic groundedness, citation and abstention metrics
apply to real model answers unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

RAG_SYSTEM_PROMPT = (
    "You are a customer-support assistant for Northwind Cloud. Answer the "
    "user's question using ONLY the documents provided in the context. "
    "Rules: (1) End every factual sentence with a citation in the exact form "
    "[doc:<document_id>] using the ids given. (2) If the documents do not "
    "contain the answer, reply with exactly: I don't know. (3) Do not invent "
    "policies, numbers or features. (4) Answer in at most three sentences."
)
NO_DOCUMENTS = "(no documents found)"
CONTEXT_HEADER = "Context documents:\n"
QUESTION_MARKER = "\n\nQuestion: "


def rag_user_prompt(question: str, documents: Sequence[tuple[str, str]]) -> str:
    """The user message: retrieved documents, best first, then the question."""
    parts = [f"[doc:{doc_id}]\n{text}" for doc_id, text in documents]
    context = "\n\n".join(parts) if parts else NO_DOCUMENTS
    return f"{CONTEXT_HEADER}{context}{QUESTION_MARKER}{question}"
