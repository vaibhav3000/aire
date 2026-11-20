# PR comments are posted by a separate workflow_run job that never runs PR code

Status: accepted (Step 2)

A `pull_request` run from a fork gets a read-only token, so it cannot comment,
and `pull_request_target` would hand a write token to a job that builds
untrusted code. AIRE splits the work: `eval-pr.yml` evaluates the PR with a
read-only token and uploads the gate result as an artifact; `eval-pr-comment.yml`
runs on `workflow_run` with `pull-requests: write`, checks out only the default
branch, treats the artifact as untrusted data (schema-checked, every string
escaped when rendered), confirms the PR's current head commit is the commit that
was evaluated, and creates or updates a single comment identified by a hidden
marker. The cost is a second workflow and a short delay before the comment
appears.

## Amendment: a pull request that changes the evaluation set

The base gate pairs the PR's fresh run with the base branch's committed
results case by case, so it refuses (exit 2) when the PR changes the
evaluation set itself, which a PR growing the suite must do. Failing such a
PR forever would block every change to the evaluation set; weakening the gate
to compare unpaired runs would make its verdicts meaningless.

Decision: with `--report`, the gate writes a `not_comparable` report with
reason `dataset_changed`, the dataset path and both hashes, and still exits 2.
`eval-pr.yml` snapshots the PR's committed results before the demo reruns and
always gates the fresh run against that snapshot too. `aire pr-check` decides
the check: a `dataset_changed` exit 2 from the base gate is a warning
annotation, not a failure; any other exit 2, a regression in either gate, or a
gate that never ran fails it. The comment renders a "Not comparable: this PR
changes the evaluation set" line with the file, both hashes and the result of
the gate against the PR's own results.

Consequence: a PR that changes the evaluation set is not compared with the
base branch at all; only its self-consistency is checked, so a reviewer must
judge its committed results directly. The report is untrusted like any other
artifact, and the renderer escapes its strings the same way.
