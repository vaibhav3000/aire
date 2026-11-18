"""Run storage, manifests, trace schema and dataset loading."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from aire.evaluator.engine import EvaluationEngine, write_eval_results
from aire.llm.base import SUTResult, SystemUnderTest
from aire.llm.deterministic import ExtractiveRAG
from aire.llm.openai_compat import OpenAIChatResponder
from aire.llm.provider import OpenAICompatProvider
from aire.runner.dataset import EvalCase, load_eval_dataset
from aire.runner.runner import ExperimentRunner, RunnerConfig
from aire.schema import Trace
from aire.tracing.recorder import TraceRecorder
from aire.tracing.storage import RunStore

CORPUS = {"doc_a": "The refund window is 30 days for annual plans."}


def _write_dataset(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _case(case_id: str) -> dict[str, Any]:
    return {"case_id": case_id, "input": "How long is the refund window?", "must_cite": False}


# ---------------------------------------------------------------- crash safety


def test_traces_are_on_disk_as_soon_as_they_are_appended(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "run1", "demo")
    store.append_trace(Trace(run_id="run1", case_id="c1", system_config={}))
    lines = (tmp_path / "run1" / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


class _BrokenUsageSUT(SystemUnderTest):
    """Answers once, then reports an invalid (negative) token count."""

    name = "broken"

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, case_input: str, case_context: dict[str, Any] | None = None) -> SUTResult:
        self.calls += 1
        tokens = 5 if self.calls == 1 else -1
        return SUTResult(
            answer="ok",
            usage={"prompt_tokens": tokens, "completion_tokens": 1, "token_source": "api"},
        )


def test_a_run_that_aborts_keeps_the_traces_it_completed(tmp_path: Path) -> None:
    ds = _write_dataset(tmp_path / "ds.jsonl", [_case("c1"), _case("c2"), _case("c3")])
    runner = ExperimentRunner(_BrokenUsageSUT(), RunnerConfig(timeout_s=5))
    with pytest.raises(ValueError, match="non-negative"):
        runner.run(ds, runs_root=tmp_path / "runs", run_name="r")
    (run_dir,) = (tmp_path / "runs").iterdir()
    lines = (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["case_id"] for line in lines] == ["c1"]


def test_a_failed_rewrite_leaves_the_previous_results_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_eval_results(tmp_path, {"version": 1})

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        write_eval_results(tmp_path, {"version": 2})
    path = tmp_path / "eval_results.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["eval_results.json"]


# ---------------------------------------------------------------- trace schema


def test_each_trace_line_carries_a_schema_version(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "run1", "demo")
    store.append_trace(Trace(run_id="run1", case_id="c1", system_config={}))
    line = (tmp_path / "run1" / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(line)["schema_version"] == "1.1"


def test_reader_ignores_fields_it_does_not_know() -> None:
    data = Trace(run_id="r", case_id="c", system_config={}).to_dict()
    data["spans"] = [
        {"kind": "generation", "name": "g", "started_at": 0.0, "latency_ms": 2.5, "model": "x"}
    ]
    data["usage"]["model"] = "x"
    data["added_later"] = True
    trace = Trace.from_dict(data)
    assert trace.spans[0].latency_ms == 2.5


def test_recorder_rejects_tool_calls_that_are_not_objects() -> None:
    recorder = TraceRecorder("r", "c", {})
    malformed: Any = [["search", {"q": 1}]]
    with pytest.raises(ValueError, match="tool_calls"):
        recorder.finish("answer", tool_calls=malformed)


# ---------------------------------------------------------------- case identity


def test_store_rejects_a_second_trace_for_the_same_case(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "run1", "demo")
    store.append_trace(Trace(run_id="run1", case_id="c1", system_config={}))
    with pytest.raises(ValueError, match="duplicate"):
        store.append_trace(Trace(run_id="run1", case_id="c1", system_config={}))


def test_dataset_with_duplicate_case_ids_is_rejected(tmp_path: Path) -> None:
    ds = _write_dataset(tmp_path / "ds.jsonl", [_case("c1"), _case("c1")])
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_dataset(ds)


def test_evaluate_run_rejects_duplicate_traces() -> None:
    case = EvalCase(case_id="c1", input="q", must_cite=False)
    traces = [
        Trace(run_id="r", case_id="c1", system_config={}, final_answer="yes"),
        Trace(run_id="r", case_id="c1", system_config={}, final_answer="no"),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        EvaluationEngine().evaluate_run(traces, [case])


def test_evaluate_run_rejects_traces_for_unknown_cases() -> None:
    case = EvalCase(case_id="c1", input="q", must_cite=False)
    traces = [
        Trace(run_id="r", case_id="c1", system_config={}, final_answer="yes"),
        Trace(run_id="r", case_id="c9", system_config={}, final_answer="yes"),
    ]
    with pytest.raises(ValueError, match="c9"):
        EvaluationEngine().evaluate_run(traces, [case])


# ---------------------------------------------------------------- dataset loader


def test_misspelled_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="expect_abstension"):
        EvalCase.from_dict({"case_id": "c", "input": "q", "expect_abstension": True})


def test_string_flag_is_rejected() -> None:
    with pytest.raises(ValueError, match="expect_abstention"):
        EvalCase.from_dict({"case_id": "c", "input": "q", "expect_abstention": "false"})


def test_keywords_must_be_a_list_of_strings() -> None:
    with pytest.raises(ValueError, match="expected_keywords"):
        EvalCase.from_dict({"case_id": "c", "input": "q", "expected_keywords": "30 days"})


def test_loader_reports_the_offending_line(tmp_path: Path) -> None:
    rows = [_case("c1"), {**_case("c2"), "must_cite": "no"}]
    ds = _write_dataset(tmp_path / "ds.jsonl", rows)
    with pytest.raises(ValueError, match="line 2"):
        load_eval_dataset(ds)


# ---------------------------------------------------------------- manifest


def test_manifest_records_a_portable_dataset_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    ds = _write_dataset(root / "demo" / "eval.jsonl", [_case("c1")])
    runner = ExperimentRunner(ExtractiveRAG("t", CORPUS, top_k=1), RunnerConfig(timeout_s=5))
    run_dir = runner.run(ds, runs_root=tmp_path / "runs", run_name="r", path_root=root)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_path"] == "demo/eval.jsonl"


def test_dataset_outside_the_root_is_recorded_by_file_name(tmp_path: Path) -> None:
    ds = _write_dataset(tmp_path / "elsewhere" / "eval.jsonl", [_case("c1")])
    runner = ExperimentRunner(ExtractiveRAG("t", CORPUS, top_k=1), RunnerConfig(timeout_s=5))
    run_dir = runner.run(ds, runs_root=tmp_path / "runs", path_root=tmp_path / "repo")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_path"] == "eval.jsonl"


def test_manifest_hash_identifies_the_corpus(tmp_path: Path) -> None:
    ds = _write_dataset(tmp_path / "ds.jsonl", [_case("c1")])

    def corpus_hash(corpus: dict[str, str]) -> str:
        runner = ExperimentRunner(ExtractiveRAG("t", corpus, top_k=1), RunnerConfig(timeout_s=5))
        run_dir = runner.run(ds, runs_root=tmp_path / "runs", corpus=corpus)
        return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["corpus_sha256"]

    first = corpus_hash(CORPUS)
    assert len(first) == 64
    assert corpus_hash(dict(CORPUS)) == first
    assert corpus_hash({"doc_a": CORPUS["doc_a"] + " Edited."}) != first


class _NondeterministicSUT(SystemUnderTest):
    name = "sampler"
    deterministic = False

    def invoke(self, case_input: str, case_context: dict[str, Any] | None = None) -> SUTResult:
        return SUTResult(answer="I don't know.")


def test_manifest_records_whether_the_system_is_deterministic(tmp_path: Path) -> None:
    ds = _write_dataset(tmp_path / "ds.jsonl", [_case("c1")])
    runs = tmp_path / "runs"
    fixed = ExperimentRunner(ExtractiveRAG("t", CORPUS, top_k=1), RunnerConfig(timeout_s=5))
    sampled = ExperimentRunner(_NondeterministicSUT(), RunnerConfig(timeout_s=5))
    for runner, expected in ((fixed, True), (sampled, False)):
        run_dir = runner.run(ds, runs_root=runs)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["deterministic"] is expected


def test_llm_responder_is_a_nondeterministic_system_under_test() -> None:
    provider = OpenAICompatProvider(
        base_url="http://127.0.0.1:9", api_key_env="AIRE_TEST_UNUSED_KEY", model="m"
    )
    sut = OpenAIChatResponder("llm", CORPUS, provider)
    assert isinstance(sut, SystemUnderTest)
    assert sut.deterministic is False
