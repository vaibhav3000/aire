# The evaluation set

`demo/eval_set.jsonl` holds 150 cases over the twelve synthetic support
documents in `demo/corpus/` (a fictional company, Northwind Cloud). The first
26 lines are the original Step 1 suite, unchanged; the other 124 were added in
Step 2 so that the paired tests in the gate have enough cases to separate a
real change from noise, and so that abstention is measured on more than four
questions. `demo/validate_eval_set.py` enforces every rule below and runs in
CI.

## What is in it

| Category | Cases | Tags |
| --- | --- | --- |
| Answerable, one document, direct wording | 96 (20 original) | `answerable`, topic, optionally `numeric` or `condition` |
| Answerable, one document, paraphrased | 18 | `answerable`, topic, `paraphrase` |
| Answerable, two documents | 16 (2 original) | `answerable`, `multi-doc` |
| Unanswerable | 20 (4 original) | `unanswerable`, topic |

Every document is the gold source of 8 to 11 single-document cases, and
appears in 2 to 4 two-document cases.

- **Direct** questions use the vocabulary a customer would pick up from the
  docs: prices, limits, time windows, eligibility rules, exceptions,
  procedures. `numeric` marks answers that are a number or a calculation;
  `condition` marks answers that hinge on an exception or an eligibility rule.
- **Paraphrased** questions avoid the document's key terms ("how fast will
  you pick up my ticket" instead of "first response time"). They exist to
  stress keyword retrieval, which is the weakest part of the demo systems.
- **Two-document** questions need one fact from each document, and every such
  case has at least one expected keyword that occurs in only one of its two
  documents.
- **Unanswerable** questions are plausible and close to the corpus topics
  (payment methods, certifications, security features, integrations, compute
  offerings) but answered nowhere. Each was checked by reading all twelve
  documents and by searching them for the question's key terms.

## How the cases were written

1. Each document was read and its facts listed; questions were written to
   cover different facts in each document rather than one fact many ways.
2. For each answerable case, the expected keywords are the facts a correct
   answer has to contain (numbers with units, named limits, conditions), not
   words copied from the question. A checker flagged any keyword that also
   appeared in its question, and those questions were reworded.
3. Keywords must occur in the supporting documents exactly as the engine
   matches them (`aire.evaluator.metrics.keyword_in_text`: case-insensitive,
   on token boundaries), between three and five per case, all lowercase.
4. Unanswerable cases carry no keywords and no documents, set
   `must_cite: false` and `expect_abstention: true`.
5. The whole file is validated by `demo/validate_eval_set.py`: schema, known
   fields, unique ids, keyword placement, abstention format, tag consistency
   (`answerable` or `unanswerable` first; `multi-doc` exactly when two
   documents are listed) and at least 20 unanswerable cases.

No threshold, prompt or retrieval setting was tuned on these cases: every
system and engine setting used by the demo was fixed in Step 1, before the
new cases existed.

## Known weaknesses

- **Reused facts.** Each document holds about eight facts and now backs about
  twelve cases, so some cases share a source sentence while asking different
  things (for example `retention_pitr_plans` and
  `retention_rewind_after_bad_release`). Seven new cases revisit facts from
  the original suite as calculations or comparisons (`sla_credit_cap_calc`,
  `billing_starter_overage_calc`, ...).
- **Calculated answers.** A computed value (2.40 USD of overage, a 10% credit)
  does not occur in the corpus, so those cases list the inputs of the
  calculation as keywords. Keyword recall therefore checks that the answer
  quotes the right facts, not that the arithmetic is right; the rubric judge
  is the check for that.
- **One inferred answer.** `support_business_friday_calc` assumes that the
  4-hour Business target counts business hours only, which the document
  implies but does not state.
- **Generic keywords.** A few keywords (`available`, `capped`, `listed`) could
  also appear in a wrong answer; every case also has at least two specific
  keywords.
- **Borderline abstention.** `absent_wire_transfer` is the closest call: the
  refunds document mentions refunds to the original payment card, but no
  document lists accepted payment methods.
- **One author, one corpus.** The suite measures this demo, written by one
  author over a small synthetic corpus. It has no separate development split;
  if settings are ever tuned, a held-out split should be created first.
