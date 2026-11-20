"""Calibrate judges against the labelled calibration set.

Offline (the default) it scores the deterministic LexicalRubricJudge on the
60 items in items.jsonl against labels.jsonl, adds every model judge whose
verdicts were recorded in verdicts_*.jsonl, and writes
results/judge_calibration.json with raw agreement, Cohen's kappa and the
confusion table per criterion. No network.

A model judge is recorded with --live (which calls the provider and needs
the owner's approval first; run --dry-run for the number of calls and tokens):

    python demo/judge_calibration/calibrate_judges.py
    python demo/judge_calibration/calibrate_judges.py --dry-run --pairwise --provider gemini --model M
    python demo/judge_calibration/calibrate_judges.py --live --pairwise --provider gemini --model M

--live runs the RubricJudge on every item and, with --pairwise, the
PairwiseJudge on v2 against v1 and v2 against v3 for each of the 20 cases
(both orders), and writes verdicts_<model>.jsonl: a header line with the
judge identity, date and usage, then one line per verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aire.evaluator.calibration import agreement, load_labels
from aire.evaluator.judges import (
    LexicalRubricJudge,
    PairwiseJudge,
    PairwiseVerdict,
    RubricJudge,
    pairwise_summary,
)
from aire.providers.cache import ResponseCache
from aire.providers.client import ModelCallError, ModelClient, RetryPolicy
from aire.providers.factory import PROVIDERS, build_chat_model, sampling_temperature
from aire.providers.ratelimit import AdaptiveRateLimiter
from aire.runner.dataset import load_eval_dataset
from aire.tracing.storage import atomic_write_text

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PAIRS = (("v2_improved", "v1_baseline"), ("v2_improved", "v3_fluent_regression"))


def load_corpus() -> dict[str, str]:
    return {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted((ROOT / "demo" / "corpus").glob("*.md"))
    }


def load_items() -> list[dict[str, Any]]:
    lines = (HERE / "items.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _docs(corpus: dict[str, str], ids: Sequence[str]) -> list[tuple[str, str]]:
    return [(doc_id, corpus[doc_id]) for doc_id in ids if doc_id in corpus]


def _criteria(verdicts: list[tuple[bool, bool]], labels: list[tuple[bool, bool]]) -> dict[str, Any]:
    return {
        name: agreement([v[i] for v in verdicts], [h[i] for h in labels])
        for i, name in enumerate(("correct", "supported"))
    }


async def judge_items(client: ModelClient, model: str, out_dir: Path, *, pairwise: bool) -> Path:
    """Run the model judges over the calibration set and record every verdict."""
    corpus = load_corpus()
    items = load_items()
    rubric = RubricJudge(client, model, temperature=client_temperature(client))
    rows: list[dict[str, Any]] = []

    async def grade(item: dict[str, Any]) -> dict[str, Any]:
        try:
            verdict = await rubric.judge(
                item["question"],
                item["answer"] or "",
                reference=_docs(corpus, item["gold_doc_ids"]),
                retrieved=_docs(corpus, item["retrieved_doc_ids"]),
            )
        except ModelCallError as exc:
            return {
                "kind": "rubric",
                "case_id": item["case_id"],
                "version": item["version"],
                "error": str(exc)[:300],
            }
        return {
            "kind": "rubric",
            "case_id": item["case_id"],
            "version": item["version"],
            "correct": verdict.correct,
            "supported": verdict.supported,
            "rationale": verdict.rationale,
        }

    rows.extend(await asyncio.gather(*(grade(item) for item in items)))

    pair_judge = PairwiseJudge(client, model, temperature=client_temperature(client))
    if pairwise:
        by_key = {(i["case_id"], i["version"]): i for i in items}
        case_ids = sorted({i["case_id"] for i in items})

        async def compare(case_id: str, a: str, b: str) -> dict[str, Any]:
            item_a, item_b = by_key[(case_id, a)], by_key[(case_id, b)]
            try:
                verdict = await pair_judge.compare(
                    item_a["question"],
                    item_a["answer"] or "",
                    item_b["answer"] or "",
                    reference=_docs(corpus, item_a["gold_doc_ids"]),
                )
            except ModelCallError as exc:
                return {
                    "kind": "pairwise",
                    "case_id": case_id,
                    "a": a,
                    "b": b,
                    "error": str(exc)[:300],
                }
            return {
                "kind": "pairwise",
                "case_id": case_id,
                "a": a,
                "b": b,
                "winner": verdict.winner,
                "consistent": verdict.consistent,
                "a_first": verdict.a_first,
                "b_first": verdict.b_first,
            }

        rows.extend(await asyncio.gather(*(compare(c, a, b) for a, b in PAIRS for c in case_ids)))

    header = {
        "identity": rubric.identity(),
        "pairwise_identity": pair_judge.identity() if pairwise else None,
        "recorded_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "n_rubric": sum(1 for r in rows if r["kind"] == "rubric"),
        "n_pairwise": sum(1 for r in rows if r["kind"] == "pairwise"),
        "client": client.stats(),
    }
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in model)
    out = out_dir / f"verdicts_{safe}.jsonl"
    atomic_write_text(out, "".join(json.dumps(r) + "\n" for r in [header, *rows]))
    return out


def client_temperature(client: ModelClient) -> float | None:
    return None if client.provider == "anthropic" else 0.0


def calibrate(verdicts_dir: Path) -> dict[str, Any]:
    corpus = load_corpus()
    cases = {c.case_id: c for c in load_eval_dataset(ROOT / "demo" / "eval_set.jsonl")}
    items = load_items()
    labels = {(r["case_id"], r["version"]): r for r in load_labels(HERE / "labels.jsonl")}
    keys = [(i["case_id"], i["version"]) for i in items]
    human = [(bool(labels[k]["correct"]), bool(labels[k]["supported"])) for k in keys]

    lexical = LexicalRubricJudge()
    lexical_verdicts = [
        lexical.judge_answer(
            item["answer"],
            cases[item["case_id"]].expected_keywords,
            cases[item["case_id"]].expect_abstention,
            dict(_docs(corpus, item["retrieved_doc_ids"])),
        )
        for item in items
    ]
    statuses = sorted({str(r.get("status", "unknown")) for r in labels.values()})
    report: dict[str, Any] = {
        "description": (
            "Judge agreement with the labels in demo/judge_calibration/labels.jsonl over the "
            "60 items in items.jsonl (20 cases x 3 demo versions)."
        ),
        "labels": {
            "n": len(keys),
            "status": statuses[0] if len(statuses) == 1 else statuses,
            "note": "draft labels are unreviewed; kappa against them is provisional",
            "positive_rate": {
                "correct": round(sum(h[0] for h in human) / len(human), 4),
                "supported": round(sum(h[1] for h in human) / len(human), 4),
            },
        },
        "judges": {
            "rubric_lexical": {"identity": lexical.identity(), **_criteria(lexical_verdicts, human)}
        },
        "pairwise": {},
    }

    for path in sorted(verdicts_dir.glob("verdicts_*.jsonl")):
        lines = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        header, rows = lines[0], lines[1:]
        identity = header["identity"]
        graded = {
            (r["case_id"], r["version"]): r
            for r in rows
            if r["kind"] == "rubric" and "error" not in r
        }
        usable = [k for k in keys if k in graded]
        report["judges"][f"rubric_llm:{identity['model']}"] = {
            "identity": identity,
            "recorded_at_utc": header["recorded_at_utc"],
            "source": path.name,
            "n_errors": len(keys) - len(usable),
            **_criteria(
                [(bool(graded[k]["correct"]), bool(graded[k]["supported"])) for k in usable],
                [(bool(labels[k]["correct"]), bool(labels[k]["supported"])) for k in usable],
            ),
        }
        pairs: dict[str, Any] = {}
        for a, b in PAIRS:
            verdicts = [
                PairwiseVerdict(r["winner"], r["consistent"], r["a_first"], r["b_first"])
                for r in rows
                if r["kind"] == "pairwise" and r["a"] == a and r["b"] == b and "error" not in r
            ]
            if verdicts:
                pairs[f"{a}_vs_{b}"] = pairwise_summary(verdicts)
        if pairs:
            report["pairwise"][identity["model"]] = pairs
    return report


def dry_run(args: argparse.Namespace) -> None:
    from aire.providers.types import ChatRequest, Message

    corpus = load_corpus()
    items = load_items()
    prompt_chars = [
        len(i["question"])
        + len(i["answer"] or "")
        + sum(len(t) for _, t in _docs(corpus, i["gold_doc_ids"] + i["retrieved_doc_ids"]))
        for i in items
    ]
    rubric_tokens = sum(
        ChatRequest(
            model=args.model, messages=(Message("user", "x" * c),), max_output_tokens=400
        ).estimated_tokens()
        + 200  # the rubric instructions
        for c in prompt_chars
    )
    n_pairwise = 2 * len(PAIRS) * len({i["case_id"] for i in items}) if args.pairwise else 0
    pairwise_tokens = n_pairwise * (300 + 200 + max(prompt_chars) // 4)
    print(f"provider: {args.provider}")
    print(f"model: {args.model}")
    print(f"rubric calls: {len(items)}")
    print(f"pairwise calls: {n_pairwise}")
    print(
        f"estimated tokens: at most {rubric_tokens + pairwise_tokens} (upper bound, output caps included)"
    )
    print("no request was sent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate judges against the labelled set.")
    parser.add_argument("--live", action="store_true", help="call the provider (needs approval)")
    parser.add_argument(
        "--dry-run", action="store_true", help="print calls and tokens; send nothing"
    )
    parser.add_argument("--pairwise", action="store_true", help="also run the pairwise judge")
    parser.add_argument("--provider", choices=PROVIDERS, default="gemini")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--rpm", type=float, default=10.0)
    parser.add_argument("--verdicts-dir", type=Path, default=HERE)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "judge_cache")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "judge_calibration.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        dry_run(args)
        return
    if args.live:
        client = ModelClient(
            build_chat_model(args.provider, args.base_url),
            limiter=AdaptiveRateLimiter(requests_per_minute=args.rpm, tokens_per_minute=None),
            cache=ResponseCache(args.cache_dir),
            retry=RetryPolicy(max_attempts=5, base_delay_s=1.0, max_delay_s=60.0),
            rng=random.Random(0),
        )
        assert sampling_temperature(args.provider) == client_temperature(client)
        path = asyncio.run(
            judge_items(client, args.model, args.verdicts_dir, pairwise=args.pairwise)
        )
        print(f"recorded verdicts in {path}")
    report = calibrate(args.verdicts_dir)
    atomic_write_text(args.out, json.dumps(report, indent=2) + "\n")
    for name, judge in report["judges"].items():
        print(
            f"{name}: correct kappa={judge['correct']['kappa']} agreement={judge['correct']['agreement']}; "
            f"supported kappa={judge['supported']['kappa']} agreement={judge['supported']['agreement']}"
        )
    print(f"labels: {report['labels']['n']} items, status {report['labels']['status']}")


if __name__ == "__main__":
    main()
