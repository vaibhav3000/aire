"""Deterministic (offline) extractive responder used by the demo RAG system.

Why a deterministic "LLM stand-in"?
-----------------------------------
The evaluation framework is the project; the demo application only needs to be
*good enough to evaluate*. A rule-based extractive responder makes every
committed result reproducible without API keys, while the `SystemUnderTest`
interface accepts real LLM adapters (see openai_compat.py) unchanged.

The responder answers extractively: it picks sentences from the retrieved
documents that share the most keywords with the question and prefixes them
with citations of the form [doc:<id>]. `style="fluent"` (the deliberate
regression in the demo) paraphrases by stripping citations and merging
sentences — which reads better but breaks grounding, exactly what the
groundedness judge should catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .base import SUTResult, SystemUnderTest

_WORD_RE = re.compile(r"[a-z0-9']+")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "for", "on",
    "and", "or", "does", "do", "what", "how", "when", "which", "who", "can",
    "with", "my", "our", "your", "i", "we", "it", "at", "by", "from", "be",
}


def content_keywords(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1}


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float


class KeywordRetriever:
    """Transparent bag-of-words retriever over a fixed corpus.

    weighting="overlap": |query ∩ doc| / |query| (plain bag-of-words).
    weighting="idf": inverse-document-frequency weighted overlap. Words that
    appear in few documents dominate the score, and query words that appear in
    NO document count only against the score, which is what makes plausible-but-
    unanswerable questions ("do you support IPv6-only clusters?") score low.
    """

    def __init__(self, corpus: dict[str, str], top_k: int = 2, weighting: str = "overlap") -> None:
        if weighting not in ("overlap", "idf"):
            raise ValueError(f"unknown weighting {weighting!r}")
        self.corpus = corpus
        self.top_k = top_k
        self.weighting = weighting
        # Precompute per-document keyword sets and document frequencies.
        self._doc_keywords: dict[str, set[str]] = {
            doc_id: content_keywords(text) for doc_id, text in corpus.items()
        }
        n_docs = len(corpus)
        self._idf: dict[str, float] = {}
        if n_docs:
            df: dict[str, int] = {}
            for keywords in self._doc_keywords.values():
                for w in keywords:
                    df[w] = df.get(w, 0) + 1
            import math

            self._idf = {w: math.log(n_docs / max(count, 1)) for w, count in df.items()}
        self._default_idf = math.log(n_docs + 1) if n_docs else 1.0

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        q = content_keywords(query)
        if not q:
            # Degenerate query: no signal - return nothing rather than arbitrary docs.
            return []
        scored: list[RetrievedDoc] = []
        for doc_id, doc_keywords in self._doc_keywords.items():
            if self.weighting == "idf":
                weights = {w: self._idf.get(w, self._default_idf) for w in q}
                denom = sum(weights.values())
                if denom <= 0:
                    continue
                score = sum(weights[w] for w in q if w in doc_keywords) / denom
            else:
                score = len(q & doc_keywords) / max(len(q), 1)
            scored.append(RetrievedDoc(doc_id, self.corpus[doc_id], round(score, 4)))
        scored.sort(key=lambda r: r.score, reverse=True)
        # Zero-overlap docs carry no signal; returning them would let a bad
        # retrieval look like a confident answer downstream.
        return [r for r in scored if r.score > 0.0][: self.top_k]


class ExtractiveRAG(SystemUnderTest):
    """Baseline RAG under test: keyword retrieval -> extractive answer with citations.

    Config knobs (intentionally few, so demo comparisons stay interpretable):
      - top_k: number of retrieved documents
      - min_score: below this the system abstains ("I don't know.")
      - style: "faithful" quotes sentences with citations;
               "fluent" merges sentences and drops citations (grounding hazard)
      - allow_irrelevant: if True, answer from top doc even when its score is 0
    """

    def __init__(
        self,
        name: str,
        corpus: dict[str, str],
        top_k: int = 2,
        min_score: float = 0.05,
        style: str = "faithful",
        weighting: str = "overlap",
        allow_irrelevant: bool = False,
    ) -> None:
        self.name = name
        self.corpus = corpus
        self.config = {
            "top_k": top_k,
            "min_score": min_score,
            "style": style,
            "weighting": weighting,
            "allow_irrelevant": allow_irrelevant,
        }
        self.retriever = KeywordRetriever(corpus, top_k=top_k, weighting=weighting)

    def invoke(self, case_input: str, case_context: dict[str, Any] | None = None) -> SUTResult:
        retrieved = self.retriever.retrieve(case_input)
        doc_ids = [r.doc_id for r in retrieved]
        if not retrieved or (retrieved[0].score < self.config["min_score"] and not self.config["allow_irrelevant"]):
            return SUTResult(answer="I don't know.", retrieved_doc_ids=doc_ids)

        best = retrieved[0]
        q = content_keywords(case_input)
        sentences = re.split(r"(?<=[.!?])\s+", best.text.strip())
        ranked = sorted(
            (
                (len(q & content_keywords(s)) / max(len(q), 1), s.strip())
                for s in sentences
                if s.strip()
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        top_sentences = [s for score, s in ranked[:2] if score > 0]

        if self.config["style"] == "fluent":
            # Regression hazard: merge sentences into one flowing sentence, drop citations.
            merged = " ".join(s.rstrip(".") for s in top_sentences) + "."
            merged = merged.capitalize()
            return SUTResult(
                answer=merged or "I don't know.",
                retrieved_doc_ids=doc_ids,
                prompt_text=case_input,
            )

        if not top_sentences:
            return SUTResult(answer="I don't know.", retrieved_doc_ids=doc_ids)
        answer = " ".join(f"{s} [doc:{best.doc_id}]" for s in top_sentences)
        return SUTResult(answer=answer, retrieved_doc_ids=doc_ids, prompt_text=case_input)
