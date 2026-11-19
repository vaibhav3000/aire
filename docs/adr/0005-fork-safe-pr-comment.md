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
