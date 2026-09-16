"""AIRE — AI Reliability & Evaluation Engine.

Evaluates AI systems (LLM apps, RAG pipelines, tool-using agents) from
recorded traces: deterministic metrics, judges, failure classification,
run-to-run regression detection, and static HTML reports.

The defining feature is *replayable evaluation*: any system version can be
re-run over the same evaluation set, and the regression engine reports
exactly which metrics improved or degraded between versions.
"""

__version__ = "1.0.0"
