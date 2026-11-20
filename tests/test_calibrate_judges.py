"""demo/judge_calibration/calibrate_judges.py: offline calibration and the judged path."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from aire.providers.client import ModelClient
from aire.providers.fake import ScriptedChatModel
from aire.providers.types import ChatRequest, ChatResponse, TokenUsage

ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    path = ROOT / "demo" / "judge_calibration" / "calibrate_judges.py"
    spec = importlib.util.spec_from_file_location("calibrate_judges", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_offline_calibration_reports_the_lexical_judge_against_the_labels(tmp_path: Path) -> None:
    script = _load()
    out = tmp_path / "calibration.json"
    script.main(["--out", str(out), "--verdicts-dir", str(tmp_path)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["labels"]["n"] == 60
    assert report["labels"]["status"] == "draft"
    lexical = report["judges"]["rubric_lexical"]
    for criterion in ("correct", "supported"):
        assert lexical[criterion]["n"] == 60
        assert -1.0 <= lexical[criterion]["kappa"] <= 1.0
    assert sum(lexical["correct"]["confusion"].values()) == 60


def _always_correct(request: ChatRequest) -> ChatResponse:
    if "Answer A:" in request.messages[-1].content:
        payload = {"winner": "A", "rationale": "first"}
    else:
        payload = {"correct": True, "supported": True, "rationale": "fine"}
    return ChatResponse(
        text=json.dumps(payload),
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        response_model="judge",
        finish_reason="stop",
        parsed=payload,
    )


def test_judged_run_writes_verdicts_and_calibrates_them(tmp_path: Path) -> None:
    script = _load()
    client = ModelClient(ScriptedChatModel(default=_always_correct))
    path = asyncio.run(script.judge_items(client, "judge", tmp_path, pairwise=True))
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["identity"]["model"] == "judge"
    assert header["n_rubric"] == 60
    out = tmp_path / "calibration.json"
    script.main(["--out", str(out), "--verdicts-dir", str(tmp_path)])
    report = json.loads(out.read_text(encoding="utf-8"))
    judged = report["judges"]["rubric_llm:judge"]
    assert judged["correct"]["confusion"]["yes/no"] > 0  # it says yes to everything
    # The fake always picks position A, so no pairwise winner survives the swap.
    assert report["pairwise"]["judge"]["v2_improved_vs_v1_baseline"]["position_consistency"] == 0.0


def test_dry_run_counts_the_calls(capsys: pytest.CaptureFixture[str]) -> None:
    script = _load()
    script.main(["--dry-run", "--pairwise", "--provider", "gemini", "--model", "m"])
    out = capsys.readouterr().out
    assert "rubric calls: 60" in out
    assert "pairwise calls: 80" in out
    assert "estimated tokens:" in out
