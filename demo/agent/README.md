# Agent trajectory suite

Real tool-calling trajectories from the sibling project repo-engineer, scored
by aire's trajectory metrics (`src/aire/evaluator/trajectory.py`).

## Where the data comes from

- `traces/*.json`: the four benchmark traces committed in repo-engineer at
  commit `f7f38f9b467622ae6f0b77d6038de5ed4c611b46`
  (`results/benchmark_traces/<task>_trace.json`), copied unchanged. They were
  produced by repo-engineer's scripted planner, so they carry no model token
  usage, and the logs record no timings, so aire reports no latency for them.
- `cases.jsonl`: one case per task. `input` is the task description from the
  trace; `expected_tool_sequence` is the reference trajectory, written from the
  task description and the fixture repository, not from the recorded run. Every
  reference starts with the runtime's fixed `list_tree` (analyze) step and ends
  with its fixed `run_tests` (verify) step.
- `tool_schemas.json`: JSON Schema for each repo-engineer tool, transcribed from
  its `ToolSpec` declarations (required and optional arguments, integer ranges,
  string length caps).
- `variants/*.json`: synthetic. `make_variants.py` applies exactly one
  documented defect per task (an argument of the wrong type, a repeated search,
  a skipped test run, an undeclared argument) and records it in
  `synthetic_mutation`. They exist to show the metrics and the gate catching
  agent regressions on known defects.

## How it is scored

`demo/run_demo.py` ingests both folders with
`aire.agents.ingest.load_repo_engineer_traces`, scores them against
`cases.jsonl` and `tool_schemas.json`, and compares them
(`results/agent_regressed_vs_agent_baseline.json`). With four tasks the
comparison has too few paired cases for a statistical test, so it uses the
tolerance rule and says so.
