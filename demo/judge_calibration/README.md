# Judge calibration set

A judge is a measurement instrument, so its verdicts are compared with labels
before its numbers are trusted. This folder holds that comparison.

- `items.jsonl`: 60 answers to label, built by `make_calibration_set.py`. It
  picks 20 cases with a fixed seed (10 direct, 3 paraphrased, 3 two-document, 4
  unanswerable) and takes each one's committed answer from v1, v2 and v3. Each
  item carries the question, the answer, the gold document ids and the ids the
  system retrieved.
- `labels.jsonl`: one label per item for the two rubric criteria, with a short
  reason:
  - **correct**: the answer gives the information the question asks for and
    agrees with the gold documents; for an unanswerable question only a decline
    is correct;
  - **supported**: every factual claim in the answer appears in a retrieved
    document that the answer cites; a decline is supported.
- `calibrate_judges.py`: compares judges with the labels and writes
  `results/judge_calibration.json` (raw agreement, Cohen's kappa, confusion
  table per criterion). Offline it scores the deterministic `LexicalRubricJudge`;
  `--live` adds a model judge (see the script's docstring; it calls a provider,
  so it needs approval, and `--dry-run` prints the calls and tokens first).

## Status of the labels

Every label has `"status": "draft"`. They were written during development
against the definitions above and have **not yet been reviewed by a person**, so
every kappa computed from them is provisional. To review: read each item in
`items.jsonl` next to its label, correct `correct` or `supported` where needed,
set `"status": "reviewed"` on the lines you checked, and rerun
`python demo/judge_calibration/calibrate_judges.py`.

## What the current numbers say

Against the draft labels, the lexical judge agrees on 57 of 60 `correct`
verdicts (kappa 0.90) and on all 60 `supported` verdicts (kappa 1.0). The perfect
`supported` score is expected for these systems: their answers are verbatim
quotes of the documents they cite, which is the one situation a lexical overlap
check handles exactly. The three `correct` disagreements are all
`multi_export_before_delete`: the answers give the 24-hour purge but not how to
export, so half of the expected keywords match and the 0.5 threshold accepts an
answer that leaves half of a two-part question unanswered.
