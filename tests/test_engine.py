"""Evaluation engine: per-case scoring, aggregation and the failure taxonomy."""

import json
import os
import subprocess
import sys

_AGGREGATE_SCRIPT = """
import json
from aire.evaluator.engine import EvaluationEngine
from aire.runner.dataset import EvalCase
from aire.schema import Trace

corpus = {"doc_a": "The refund window is 30 days for annual plans."}
cases = [
    EvalCase(case_id="c1", input="refund window?", expected_keywords=["30 days"],
             supporting_doc_ids=["doc_a"]),
    EvalCase(case_id="c2", input="ipv6?", expect_abstention=True, must_cite=False),
]
traces = [
    Trace(run_id="r", case_id="c1", system_config={}, retrieved_doc_ids=["doc_a"],
          final_answer="The refund window is 30 days for annual plans [doc:doc_a]."),
    Trace(run_id="r", case_id="c2", system_config={}, final_answer="I don't know."),
]
results = EvaluationEngine(corpus=corpus).evaluate_run(traces, cases)
print(json.dumps(results["aggregates"]))
"""


def _aggregates_with_hash_seed(seed: int) -> str:
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    proc = subprocess.run(
        [sys.executable, "-c", _AGGREGATE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return proc.stdout


def test_aggregates_serialise_identically_across_hash_seeds() -> None:
    """Committed eval_results.json must not churn with Python's string hash seed."""
    outputs = {_aggregates_with_hash_seed(seed) for seed in range(6)}
    assert len(outputs) == 1, outputs
    assert json.loads(outputs.pop())["n_cases"] == 2
