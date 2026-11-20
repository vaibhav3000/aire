"""Create or update one sticky pull request comment through the GitHub REST API.

Stdlib only (urllib), so the trusted comment job installs nothing but aire.
The comment is found by a hidden marker AND by its author being the bot, so
a person who quotes the marker never has their comment overwritten. Before
writing, the PR's current head commit can be checked against the commit
that was evaluated, so a result is never posted on a PR it does not belong
to (ADR 0005). Pagination only follows links on the configured API host, so
the token is never sent elsewhere.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from ..report.markdown import MARKER

_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


class GitHubError(RuntimeError):
    pass


class HeadMismatch(GitHubError):
    """The PR has moved on since the evaluated commit; its result would be stale."""


class GitHubAPI:
    def __init__(
        self, token: str, *, api_url: str = "https://api.github.com", timeout_s: float = 30.0
    ) -> None:
        if not token:
            raise GitHubError("no GitHub token given")
        self._token = token
        self.api_url = api_url.rstrip("/")
        self.timeout_s = timeout_s

    def _send(self, method: str, url: str, body: Any = None) -> tuple[Any, str | None]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self._token}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        request.add_header("User-Agent", "aire-pr-gate")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read() or b"null")
                return payload, response.headers.get("Link")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise GitHubError(
                f"{method} {url.removeprefix(self.api_url)}: HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise GitHubError(f"{method} {url.removeprefix(self.api_url)}: {exc}") from exc

    def request(self, method: str, path: str, body: Any = None) -> Any:
        return self._send(method, f"{self.api_url}{path}", body)[0]

    def paginate(self, path: str) -> list[Any]:
        url: str | None = f"{self.api_url}{path}"
        items: list[Any] = []
        while url is not None:
            page, link = self._send("GET", url)
            if not isinstance(page, list):
                raise GitHubError(f"GET {path}: expected a list")
            items.extend(page)
            match = _NEXT_RE.search(link or "")
            url = match.group(1) if match else None
            if url is not None and not url.startswith(f"{self.api_url}/"):
                raise GitHubError(f"refusing to follow a pagination link off {self.api_url}")
        return items


def upsert_sticky_comment(
    api: GitHubAPI,
    repo: str,
    pr_number: int,
    body: str,
    *,
    marker: str = MARKER,
    expected_head_sha: str | None = None,
    bot_login: str = "github-actions[bot]",
) -> str:
    """Create the bot's marked comment, or update it; returns "created" or "updated"."""
    if not _REPO_RE.match(repo) or ".." in repo:
        raise ValueError(f"not a repository name: {repo!r}")
    if not isinstance(pr_number, int) or pr_number <= 0:
        raise ValueError(f"not a pull request number: {pr_number!r}")
    if marker not in body:
        body = f"{marker}\n{body}"
    if expected_head_sha is not None:
        pull = api.request("GET", f"/repos/{repo}/pulls/{pr_number}")
        head = (pull or {}).get("head", {}).get("sha")
        if head != expected_head_sha:
            raise HeadMismatch(
                f"PR #{pr_number} head is {head}, the evaluated commit was {expected_head_sha}"
            )
    comments = api.paginate(f"/repos/{repo}/issues/{pr_number}/comments?per_page=100")
    mine = [
        c
        for c in comments
        if marker in str(c.get("body", ""))
        and c.get("user", {}).get("type") == "Bot"
        and c.get("user", {}).get("login") == bot_login
    ]
    if mine:
        api.request("PATCH", f"/repos/{repo}/issues/comments/{int(mine[-1]['id'])}", {"body": body})
        return "updated"
    api.request("POST", f"/repos/{repo}/issues/{pr_number}/comments", {"body": body})
    return "created"
