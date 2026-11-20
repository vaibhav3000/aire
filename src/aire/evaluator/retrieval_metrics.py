"""Retrieval quality metrics: recall@k, MRR.

Ground truth is declarative: each eval case lists the doc ids that SHOULD be
retrieved (supporting_doc_ids). Metrics compare the trace's retrieved_doc_ids
against that list. Simple by design: retrieval ground truth for real systems
is the hard part, and this project does not pretend otherwise.

Repeated ids in a retrieved list count once (first occurrence wins), so a
retriever that returns the same document twice cannot push a relevant one
out of the top k.
"""

from __future__ import annotations


def dedupe(retrieved: list[str]) -> list[str]:
    """Retrieved ids with repeats removed, keeping the first occurrence."""
    return list(dict.fromkeys(retrieved))


def retrieval_recall_at_k(retrieved: list[str], relevant: list[str], k: int | None = None) -> float:
    """Fraction of relevant docs found within the first k retrieved (k defaults to all)."""
    if not relevant:
        return 1.0
    unique = dedupe(retrieved)
    top = unique[:k] if k is not None else unique
    return sum(1 for d in relevant if d in top) / len(relevant)


def retrieval_mrr(retrieved: list[str], relevant: list[str], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant doc within the first k retrieved (0 if none)."""
    relevant_set = set(relevant)
    unique = dedupe(retrieved)
    top = unique[:k] if k is not None else unique
    for rank, doc_id in enumerate(top, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0
