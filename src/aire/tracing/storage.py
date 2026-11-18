"""Run storage: every experiment run writes a self-describing directory.

    runs/<run_name>/
        manifest.json     run metadata (name, config, timestamp, dataset hash)
        traces.jsonl      one trace per line
        eval_results.json per-case + aggregate metric outputs
        regression.json   only when a comparison was requested

Crash safety: each trace is appended to traces.jsonl (and fsynced) as soon as
its case finishes, so an interrupted run keeps every completed case. Whole
files (manifest, results, reports) are written to a temporary file and moved
into place with os.replace, so readers never see a half-written file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..schema import Trace, validate_trace_dict

MANIFEST_SCHEMA_VERSION = "1.1"


@dataclass
class RunManifest:
    run_id: str
    name: str
    created_at: float
    system_config: dict[str, Any]
    deterministic: bool
    dataset_path: str  # relative to the run's path root, POSIX separators
    dataset_sha256: str
    corpus_sha256: str | None
    n_cases: int
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_sha256(corpus: Mapping[str, str]) -> str:
    """Content hash of a {doc_id: text} corpus, independent of dict order."""
    canonical = json.dumps(sorted(corpus.items()), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def portable_path(path: str | Path, root: str | Path | None = None) -> str:
    """`path` relative to `root` (default: the working directory) with POSIX
    separators, or just its file name when it lies outside `root`.

    Keeps machine-specific absolute paths (and user names) out of committed
    manifests.
    """
    resolved = Path(path).resolve()
    base = Path(root).resolve() if root is not None else Path.cwd().resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return resolved.name


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write `text` to `path` through a temporary file and os.replace.

    Readers see either the previous file or the complete new one, and a
    failed write leaves the previous file untouched. Output uses LF line
    endings on every platform.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


class RunStore:
    """Creates run directories and appends validated traces as they arrive."""

    def __init__(self, runs_root: str | Path, run_id: str, name: str, fsync: bool = True) -> None:
        self.run_dir = Path(runs_root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.name = name
        self._fsync = fsync
        self._traces_path = self.run_dir / "traces.jsonl"
        self._traces_path.write_bytes(b"")
        self._traces: list[Trace] = []
        self._case_ids: set[str] = set()

    def write_manifest(
        self,
        created_at: float,
        system_config: Mapping[str, Any],
        dataset_path: str | Path,
        n_cases: int,
        *,
        deterministic: bool = True,
        corpus: Mapping[str, str] | None = None,
        path_root: str | Path | None = None,
    ) -> None:
        manifest = RunManifest(
            run_id=self.run_id,
            name=self.name,
            created_at=created_at,
            system_config=dict(system_config),
            deterministic=deterministic,
            dataset_path=portable_path(dataset_path, path_root),
            dataset_sha256=file_sha256(dataset_path),
            corpus_sha256=corpus_sha256(corpus) if corpus is not None else None,
            n_cases=n_cases,
        )
        atomic_write_text(self.run_dir / "manifest.json", json.dumps(manifest.to_dict(), indent=2))

    def append_trace(self, trace: Trace) -> None:
        """Validate one trace and append it to traces.jsonl immediately."""
        d = trace.to_dict()
        problems = validate_trace_dict(d)
        if problems:
            raise ValueError(f"invalid trace for case {trace.case_id}: {problems}")
        if trace.case_id in self._case_ids:
            raise ValueError(f"duplicate trace for case {trace.case_id!r} in run {self.run_id}")
        with open(self._traces_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(d) + "\n")
            f.flush()
            if self._fsync:
                os.fsync(f.fileno())
        self._case_ids.add(trace.case_id)
        self._traces.append(trace)

    def flush_traces(self) -> None:
        """Kept for API compatibility: append_trace already writes each trace."""

    @property
    def traces(self) -> list[Trace]:
        return list(self._traces)


def load_run_traces(run_dir: str | Path) -> list[Trace]:
    """Read traces.jsonl back as Trace objects."""
    path = Path(run_dir) / "traces.jsonl"
    traces: list[Trace] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(Trace.from_dict(json.loads(line)))
    return traces
