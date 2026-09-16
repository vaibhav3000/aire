"""Retrieval quality metrics: recall@k, MRR.

Ground truth is declarative: each eval case lists the doc ids that SHOULD be
retrieved (supporting_doc_ids). Metrics compare the trace's retrieved_doc_ids
against that list. Simple by design — retrieval ground truth for real systems
is the hard part, and this project does not pretend otherwise.
"""

from __future__ import annotations


def retrieval_recall_at_k(retrieved: list[str], relevant: list[str], k: int | None = None) -> float:
    """Fraction of relevant docs found within the first k retrieved (k defaults to all)."""
    if not relevant:
        return 1.0
    top = retrieved[:k] if k is not None else retrieved
    return sum(1 for d in relevant if d in top) / len(relevant)


def retrieval_mrr(retrieved: list[str], relevant: list[str]) -> float:
    """Mean reciprocal rank of the first relevant retrieved doc (0 if none)."""
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0
