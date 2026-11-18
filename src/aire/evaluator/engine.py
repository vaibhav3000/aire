"""Evaluation engine: traces + cases -> per-case results, aggregates, failure taxonomy.

The engine reads recorded traces (it never re-runs the system), applies the
metric suite defined by each case's expectations, classifies failures, and
writes eval_results.json. This separation (run once, evaluate many times,
compare across runs) is what makes evaluation replayable.

Scoring rules that keep a broken run from looking good:
- A failed invocation (an error, a timeout, or no answer at all) scores 0 on
  every answer-quality metric the case defines and is never a correct
  abstention. It counts in `errors` and against `success_rate`.
- Every mean in the aggregates sits next to n_<metric>, the number of cases
  it averages, so a mean over fewer cases is visible as such.
- Latency, token and cost summaries cover successful invocations only, and
  token means skip cases whose token count is unknown (token_source "none").
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..runner.dataset import EvalCase
from ..schema import Trace
from ..tracing.storage import atomic_write_text
from . import metrics as M
from . import retrieval_metrics as R
from . import tool_metrics as T

# Pricing table (USD per 1k tokens). The demo uses heuristic tokens at $0.0/1k.
DEFAULT_PRICES: dict[str, float] = {"default": 0.0}

_RECALL_KEY_RE = re.compile(r"^retrieval_recall_at_(\d+)$")
_LATENCY_KEYS = frozenset({"latency_ms"})
_TOKEN_KEYS = frozenset({"total_tokens", "cost_usd"})


def invocation_failed(trace: Trace) -> bool:
    """True when the call failed (error or timeout) or returned no answer at all."""
    return bool(trace.error) or not (trace.final_answer or "").strip()


def _deepest_recall(case_result: dict[str, Any]) -> float | None:
    """Recall at the largest configured cut-off, if the case was scored for retrieval."""
    best_k, value = -1, None
    for key, v in case_result.items():
        match = _RECALL_KEY_RE.match(key)
        if match and isinstance(v, int | float) and int(match.group(1)) > best_k:
            best_k, value = int(match.group(1)), float(v)
    return value


def _tool_failed(case_result: dict[str, Any]) -> bool:
    if case_result.get("tool_selection_correct") == 0.0:
        return True
    for key in ("tool_args_match", "tool_args_valid"):
        value = case_result.get(key)
        if isinstance(value, int | float) and value < 1.0:
            return True
    return False


def classify_failure(
    trace: Trace,
    case: EvalCase,
    case_result: dict[str, Any],
    groundedness_threshold: float = 0.5,
    keyword_threshold: float = 0.5,
) -> str | None:
    """Map a case result to one failure category from a small, documented taxonomy.

    The checks follow the pipeline, upstream stages first, so each failure is
    blamed on the earliest stage that went wrong (first match wins):
      invocation_error  - the call failed (exception, timeout, provider error)
                          or returned no answer
      missed_abstention - answered an unanswerable question
      retrieval_miss    - a gold document is missing from the top-K retrieved
                          documents (K = the largest configured recall cut-off)
      tool_error        - wrong tool, wrong arguments, or a schema-invalid call
      wrong_abstention  - abstained although the gold documents were retrieved
      ungrounded        - groundedness below threshold
      incomplete_answer - keyword recall below threshold
      answer_mismatch   - the exact-match expectation was not met
      None              - success
    """
    if invocation_failed(trace):
        return "invocation_error"
    answer_abstained = M.abstained(trace.final_answer)
    if case.expect_abstention:
        return None if answer_abstained else "missed_abstention"
    recall = _deepest_recall(case_result)
    if recall is not None and recall < 1.0:
        return "retrieval_miss"
    if _tool_failed(case_result):
        return "tool_error"
    if answer_abstained:
        return "wrong_abstention"
    gr = case_result.get("groundedness")
    if gr is not None and gr < groundedness_threshold:
        return "ungrounded"
    kr = case_result.get("keyword_recall")
    if case.expected_keywords and kr is not None and kr < keyword_threshold:
        return "incomplete_answer"
    if case_result.get("exact_match") == 0.0:
        return "answer_mismatch"
    return None


@dataclass
class EngineConfig:
    groundedness_threshold: float = 0.5
    keyword_threshold: float = 0.5
    price_per_1k: float = 0.0
    # Share of a claim's content keywords that must occur in a cited, retrieved
    # document for the claim to count as supported (metrics.groundedness).
    groundedness_min_overlap: float = M.DEFAULT_MIN_OVERLAP
    # Fixed retrieval cut-offs, so runs that retrieve different numbers of
    # documents are compared at the same K.
    recall_ks: tuple[int, ...] = (1, 3)
    mrr_k: int = 3


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 1]) of an ascending list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


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
        failed = invocation_failed(trace)
        answer_abstained = not failed and M.abstained(answer)
        result: dict[str, Any] = {
            "case_id": case.case_id,
            "trace_id": trace.trace_id,
            "answer": answer,
            "error": trace.error,
            "errors": int(failed),
        }

        # --- answer quality (only what the case specifies) ---
        if case.expected_keywords:
            result["keyword_recall"] = (
                0.0 if failed else round(M.keyword_recall(answer, case.expected_keywords), 4)
            )
        if case.expected_answer is not None:
            result["exact_match"] = 0.0 if failed else M.exact_match(answer, case.expected_answer)
        if case.expect_abstention:
            result["abstention_correct"] = float(answer_abstained)
            result["abstention_accuracy"] = float(answer_abstained)
        else:
            result["abstention_correct"] = None
            result["abstention_accuracy"] = 0.0 if failed else float(not answer_abstained)
            result["false_refusal"] = None if failed else float(answer_abstained)

        # --- grounding / citations ---
        if case.must_cite and not case.expect_abstention:
            if failed:
                result["gold_citation_recall"] = 0.0
                result["citation_precision"] = 0.0
            else:
                result["gold_citation_recall"] = round(
                    M.gold_citation_recall(answer, case.supporting_doc_ids), 4
                )
                allowed = set(case.supporting_doc_ids) | set(trace.retrieved_doc_ids)
                cp = M.citation_precision(answer, allowed)
                result["citation_precision"] = None if cp is None else round(cp, 4)
            if self.corpus:
                if failed:
                    result["groundedness"] = 0.0
                else:
                    context = {
                        doc_id: self.corpus[doc_id]
                        for doc_id in trace.retrieved_doc_ids
                        if doc_id in self.corpus
                    }
                    gr = M.groundedness(
                        answer, context, min_overlap=self.config.groundedness_min_overlap
                    )
                    result["groundedness"] = None if gr is None else round(gr, 4)

        # --- retrieval, at fixed cut-offs; an empty retrieval scores 0 ---
        if case.supporting_doc_ids:
            for k in self.config.recall_ks:
                result[f"retrieval_recall_at_{k}"] = round(
                    R.retrieval_recall_at_k(trace.retrieved_doc_ids, case.supporting_doc_ids, k), 4
                )
            k = self.config.mrr_k
            result[f"retrieval_mrr_at_{k}"] = round(
                R.retrieval_mrr(trace.retrieved_doc_ids, case.supporting_doc_ids, k), 4
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
        result["token_source"] = trace.usage.token_source
        result["cost_usd"] = M.cost_estimate(trace.usage.total_tokens, self.config.price_per_1k)

        # --- failure classification ---
        result["failure"] = classify_failure(
            trace,
            case,
            result,
            groundedness_threshold=self.config.groundedness_threshold,
            keyword_threshold=self.config.keyword_threshold,
        )
        return result

    def evaluate_run(self, traces: list[Trace], cases: list[EvalCase]) -> dict[str, Any]:
        case_ids = [c.case_id for c in cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("duplicate case_id in the evaluation cases")
        by_case: dict[str, Trace] = {}
        for t in traces:
            if t.case_id in by_case:
                raise ValueError(f"duplicate traces for case {t.case_id!r}")
            by_case[t.case_id] = t
        unknown = sorted(set(by_case) - set(case_ids))
        if unknown:
            raise ValueError(f"traces for case ids that are not in the dataset: {unknown}")
        per_case = []
        missing = [c.case_id for c in cases if c.case_id not in by_case]
        for case in cases:
            trace = by_case.get(case.case_id)
            if trace is None:
                # Score a missing trace exactly like a failed invocation.
                placeholder = Trace(
                    run_id="missing",
                    case_id=case.case_id,
                    system_config={},
                    error="missing trace",
                    trace_id="missing",
                )
                row = self.evaluate_case(placeholder, case)
                row["trace_id"] = None
                per_case.append(row)
                continue
            per_case.append(self.evaluate_case(trace, case))

        aggregates = self._aggregate(per_case)
        return {
            "per_case": per_case,
            "aggregates": aggregates,
            "missing_traces": missing,
            "engine_config": asdict(self.config),
        }

    def _aggregate(self, per_case: list[dict[str, Any]]) -> dict[str, Any]:
        """Mean and n for every metric present in at least one case result."""
        keys: set[str] = set()
        for row in per_case:
            keys.update(k for k, v in row.items() if _is_number(v))
        succeeded = [row for row in per_case if not row.get("errors")]
        token_rows = [row for row in succeeded if row.get("token_source") not in (None, "none")]

        aggregates: dict[str, Any] = {}
        for key in sorted(keys):
            if key in _LATENCY_KEYS:
                rows = succeeded
            elif key in _TOKEN_KEYS:
                rows = token_rows
            else:
                rows = per_case
            values = [row[key] for row in rows if _is_number(row.get(key))]
            if not values:
                continue
            aggregates[f"mean_{key}"] = round(statistics.fmean(values), 4)
            aggregates[f"n_{key}"] = len(values)

        latencies = sorted(
            row["latency_ms"] for row in succeeded if _is_number(row.get("latency_ms"))
        )
        if latencies:
            aggregates["p50_latency_ms"] = round(_percentile(latencies, 0.50), 4)
            aggregates["p95_latency_ms"] = round(_percentile(latencies, 0.95), 4)
        aggregates["token_sources"] = sorted({str(row["token_source"]) for row in token_rows})

        failures: dict[str, int] = {}
        for row in per_case:
            f = row.get("failure")
            if f:
                failures[f] = failures.get(f, 0) + 1
        n_success = sum(1 for row in per_case if not row.get("failure"))
        aggregates["failure_counts"] = dict(sorted(failures.items()))
        aggregates["n_cases"] = len(per_case)
        aggregates["n_success"] = n_success
        aggregates["success_rate"] = round(n_success / len(per_case), 4) if per_case else 0.0
        return aggregates


def write_eval_results(run_dir: str | Path, results: dict[str, Any]) -> Path:
    """Write eval_results.json atomically (see tracing.storage.atomic_write_text)."""
    return atomic_write_text(Path(run_dir) / "eval_results.json", json.dumps(results, indent=2))
