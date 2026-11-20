"""Build the judge calibration set: a stratified sample of committed answers.

Picks 20 cases with a fixed seed (10 direct, 3 paraphrased, 3 two-document,
4 unanswerable) and takes each one's answer from all three committed demo
versions, 60 items in all. Every item carries what a labeller (or a judge)
needs: the question, the answer, the gold documents and the documents the
system retrieved. Writes demo/judge_calibration/items.jsonl.

    python demo/judge_calibration/make_calibration_set.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VERSIONS = ("v1_baseline", "v2_improved", "v3_fluent_regression")
STRATA = {"direct": 10, "paraphrase": 3, "multi-doc": 3, "unanswerable": 4}
SEED = 7


def stratum(case: dict[str, Any]) -> str:
    if case.get("expect_abstention"):
        return "unanswerable"
    if len(case.get("supporting_doc_ids", [])) == 2:
        return "multi-doc"
    return "paraphrase" if "paraphrase" in case.get("tags", []) else "direct"


def main() -> None:
    lines = (ROOT / "demo" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]
    rng = random.Random(SEED)
    chosen: list[dict[str, Any]] = []
    for name, count in STRATA.items():
        pool = sorted((c for c in cases if stratum(c) == name), key=lambda c: c["case_id"])
        chosen.extend(rng.sample(pool, count))
    chosen.sort(key=lambda c: c["case_id"])

    items = []
    for version in VERSIONS:
        traces = {
            t["case_id"]: t
            for t in (
                json.loads(line)
                for line in (ROOT / "results" / version / "traces.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
        }
        for case in chosen:
            trace = traces[case["case_id"]]
            items.append(
                {
                    "case_id": case["case_id"],
                    "version": version,
                    "stratum": stratum(case),
                    "question": case["input"],
                    "answer": trace["final_answer"],
                    "gold_doc_ids": case.get("supporting_doc_ids", []),
                    "retrieved_doc_ids": trace["retrieved_doc_ids"],
                }
            )
    out = HERE / "items.jsonl"
    out.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {len(items)} items for {len(chosen)} cases to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
