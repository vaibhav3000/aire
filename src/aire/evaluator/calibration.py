"""Agreement between a judge and reference labels (Cohen's kappa).

A judge is a measurement instrument; before its numbers are trusted, its
verdicts are compared with labels on a calibration set. agreement() reports
raw agreement, Cohen's kappa (agreement corrected for chance) and the
confusion table, keyed "<judge>/<label>".
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .stats import cohens_kappa


def agreement(judge: Sequence[bool], labels: Sequence[bool]) -> dict[str, Any]:
    if len(judge) != len(labels):
        raise ValueError("judge verdicts and labels must cover the same items")
    pairs = list(zip(judge, labels, strict=True))
    name = {True: "yes", False: "no"}
    confusion = {f"{name[j]}/{name[h]}": 0 for j in (True, False) for h in (True, False)}
    for j, h in pairs:
        confusion[f"{name[j]}/{name[h]}"] += 1
    n = len(pairs)
    return {
        "n": n,
        "agreement": round(sum(1 for j, h in pairs if j == h) / n, 4) if n else None,
        "kappa": round(cohens_kappa(list(judge), list(labels)), 4) if n else None,
        "confusion": confusion,
    }


def load_labels(path: str | Path) -> list[dict[str, Any]]:
    """Calibration labels, one JSON object per line."""
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("case_id", "version", "correct", "supported"):
            if key not in row:
                raise ValueError(f"{path}:{line_no}: missing {key!r}")
        rows.append(row)
    return rows
