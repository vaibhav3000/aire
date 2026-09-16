"""Run storage: every experiment run writes a self-describing directory.

    runs/<run_name>/
        manifest.json     run metadata (name, config, timestamp, dataset hash)
        traces.jsonl      one trace per line
        eval_results.json per-case + aggregate metric outputs
        regression.json   only when a comparison was requested
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..schema import Trace, validate_trace_dict


@dataclass
class RunManifest:
    run_id: str
    name: str
    created_at: float
    system_config: dict[str, Any]
    dataset_path: str
    dataset_sha256: str
    n_cases: int
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class RunStore:
    """Creates run directories and appends validated traces."""

    def __init__(self, runs_root: str | Path, run_id: str, name: str) -> None:
        self.run_dir = Path(runs_root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.name = name
        self._traces_path = self.run_dir / "traces.jsonl"
        self._traces: list[Trace] = []

    def write_manifest(
        self,
        created_at: float,
        system_config: dict[str, Any],
        dataset_path: str | Path,
        n_cases: int,
    ) -> None:
        manifest = RunManifest(
            run_id=self.run_id,
            name=self.name,
            created_at=created_at,
            system_config=system_config,
            dataset_path=str(dataset_path),
            dataset_sha256=file_sha256(dataset_path),
            n_cases=n_cases,
        )
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )

    def append_trace(self, trace: Trace) -> None:
        d = trace.to_dict()
        problems = validate_trace_dict(d)
        if problems:
            raise ValueError(f"invalid trace for case {trace.case_id}: {problems}")
        self._traces.append(trace)

    def flush_traces(self) -> None:
        with open(self._traces_path, "w", encoding="utf-8") as f:
            for trace in self._traces:
                f.write(json.dumps(trace.to_dict()) + "\n")

    @property
    def traces(self) -> list[Trace]:
        return list(self._traces)


def load_run_traces(run_dir: str | Path) -> list[Trace]:
    """Read traces.jsonl back as Trace objects."""
    path = Path(run_dir) / "traces.jsonl"
    traces: list[Trace] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(Trace.from_dict(json.loads(line)))
    return traces
