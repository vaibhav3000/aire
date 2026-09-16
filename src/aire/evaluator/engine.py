"""Evaluation engine: traces + cases -> per-case results, aggregates, failure taxonomy.

The engine reads recorded traces (it never re-runs the system), applies the
metric suite defined by each case's expectations, classifies failures, and
writes eval_results.json. This separation — run once, evaluate many times,
compare across runs — is what makes evaluation replayable.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..runner.dataset import EvalCase
from ..schema import Trace
from . import metrics as M
from . import retrieval_metrics as R
from . import tool_metrics as T

# Pricing table (USD per 1k tokens) — demo uses heuristic tokens at $0.0/1k.
DEFAULT_PRICES: dict[str, float] = {"default": 0.0}


def classify_failure(trace: Trace, case: EvalCase, case_result: dict[str, Any]) -> str | None:
    """Map a case result to a failure category from a small, documented taxonomy.

    Taxonomy (mutually exclusive, first match wins):
      invocation_error  - the system call itself failed (exception/timeout)
      wrong_abstention  - abstained on an answerable question
      missed_abstention - answered an unanswerable question
      ungrounded        - groundedness below threshold despite answering
      incomplete_answer - keyword recall below threshold
      retrieval_miss    - supporting docs not retrieved
      none              - success
    """
    if trace.error:
        return "invocation_error"
    if case.expect_abstention:
        if not M.abstained(trace.final_answer):
            return "missed_abstention"
        return None
    if M.abstained(trace.final_answer):
        return "wrong_abstention"
    gr = case_result.get("groundedness")
    if gr is not None and gr < 0.5:
        return "ungrounded"
    kr = case_result.get("keyword_recall")
    if case.expected_keywords and kr is not None and kr < 0.5:
        return "incomplete_answer"
    if case.supporting_doc_ids:
        recall = case_result.get("retrieval_recall")
        if recall is not None and recall < 1.0:
            return "retrieval_miss"
    return None


@dataclass
class EngineConfig:
    groundedness_threshold: float = 0.5
    keyword_threshold: float = 0.5
    price_per_1k: float = 0.0


class EvaluationEngine:
    """Applies the metric suite to a run's traces."""

    def __init__(
        self,
        corpus: dict[str, str] | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.corpus = corpus or {}
        self.config = config or EngineConfig()

    def evaluate_case(self, trace: Trace, case: EvalCase) -> dict[str, Any]:
        answer = trace.final_answer
        result: dict[str, Any] = {
            "case_id": case.case_id,
            "trace_id": trace.trace_id,
            "answer": answer,
            "error": trace.error,
        }

        # --- quality metrics (only when the case specifies expectations) ---
        if case.expected_keywords:
            result["keyword_recall"] = round(
                M.keyword_recall(answer, case.expected_keywords), 4
            )
        if case.expected_answer is not None:
            result["exact_match"] = M.exact_match(answer, case.expected_answer)
        if case.expect_abstention:
            result["abstention_correct"] = float(M.abstained(answer))
        else:
            result["abstention_correct"] = None

        # --- grounding / citations ---
        if self.corpus and case.must_cite and not case.expect_abstention:
            result["citation_coverage"] = round(
                M.citation_coverage(answer, case.supporting_doc_ids), 4
            )
            gr = M.groundedness(answer, self.corpus)
            result["groundedness"] = round(gr, 4) if gr is not None else None

        # --- retrieval metrics ---
        if case.supporting_doc_ids and trace.retrieved_doc_ids:
            result["retrieval_recall"] = round(
                R.retrieval_recall_at_k(trace.retrieved_doc_ids, case.supporting_doc_ids), 4
            )
            result["retrieval_mrr"] = round(
                R.retrieval_mrr(trace.retrieved_doc_ids, case.supporting_doc_ids), 4
            )

        # --- tool metrics ---
        ts = T.tool_selection_correct(trace.tool_calls, case.expected_tool)
        if ts is not None:
            result["tool_selection_correct"] = ts
        if trace.tool_calls:
            result["tool_args_valid"] = round(T.tool_args_valid(trace.tool_calls), 4)
        am = T.tool_args_match(trace.tool_calls, case.expected_tool, case.expected_tool_args)
        if am is not None:
            result["tool_args_match"] = round(am, 4)

        # --- cost / latency ---
        latency = sum(s.latency_ms for s in trace.spans)
        result["latency_ms"] = round(latency, 2)
        result["total_tokens"] = trace.usage.total_tokens
        result["cost_usd"] = M.cost_estimate(
            trace.usage.total_tokens, self.config.price_per_1k
        )

        # --- failure classification ---
        result["failure"] = classify_failure(trace, case, result)
        return result

    def evaluate_run(
        self, traces: list[Trace], cases: list[EvalCase]
    ) -> dict[str, Any]:
        by_case = {t.case_id: t for t in traces}
        per_case = []
        missing = [c.case_id for c in cases if c.case_id not in by_case]
        for case in cases:
            trace = by_case.get(case.case_id)
            if trace is None:
                per_case.append(
                    {
                        "case_id": case.case_id,
                        "trace_id": None,
                        "answer": None,
                        "error": "missing trace",
                        "failure": "invocation_error",
                    }
                )
                continue
            per_case.append(self.evaluate_case(trace, case))

        aggregates = self._aggregate(per_case)
        return {
            "per_case": per_case,
            "aggregates": aggregates,
            "missing_traces": missing,
            "engine_config": {
                "groundedness_threshold": self.config.groundedness_threshold,
                "keyword_threshold": self.config.keyword_threshold,
                "price_per_1k": self.config.price_per_1k,
            },
        }

    def _aggregate(self, per_case: list[dict[str, Any]]) -> dict[str, Any]:
        """Mean over cases for every metric present in at least one case result."""
        keys: set[str] = set()
        for row in per_case:
            keys.update(k for k, v in row.items() if isinstance(v, (int, float)) and k != "case_id")
        aggregates: dict[str, Any] = {}
        for key in keys:
            values = [
                row[key] for row in per_case if isinstance(row.get(key), (int, float))
            ]
            if not values:
                continue
            aggregates[f"mean_{key}"] = round(statistics.fmean(values), 4)
        failures: dict[str, int] = {}
        for row in per_case:
            f = row.get("failure")
            if f:
                failures[f] = failures.get(f, 0) + 1
        aggregates["failure_counts"] = failures
        aggregates["n_cases"] = len(per_case)
        aggregates["n_success"] = sum(1 for row in per_case if not row.get("failure"))
        return aggregates


def write_eval_results(run_dir: str | Path, results: dict[str, Any]) -> Path:
    import json

    path = Path(run_dir) / "eval_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path
