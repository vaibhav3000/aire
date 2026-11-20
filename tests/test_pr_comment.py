"""PR comment: markdown from untrusted gate JSON, and the sticky comment upsert.

The GitHub API is a fake on 127.0.0.1; the token is a dummy.
"""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from aire.ci.github import GitHubAPI, GitHubError, HeadMismatch, upsert_sticky_comment
from aire.report.markdown import MARKER, render_gate_markdown

ROOT = Path(__file__).resolve().parents[1]


def _gate(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def test_markdown_leads_with_the_decision_and_lists_what_moved() -> None:
    text = render_gate_markdown(_gate("v3_fluent_regression_vs_v1_baseline.json"), title="v3")
    assert text.startswith(MARKER)
    assert "Blocked: 4 gated metrics regressed" in text
    assert "| `groundedness` |" in text
    assert "McNemar" in text
    assert "<details>" in text  # the full table is folded away


def test_a_passing_gate_says_so() -> None:
    text = render_gate_markdown(_gate("v2_improved_vs_v1_baseline.json"), title="v2")
    assert "Passed: no gated metric regressed" in text


def test_untrusted_strings_cannot_inject_markup_mentions_or_table_breaks() -> None:
    gate = _gate("v3_fluent_regression_vs_v1_baseline.json")
    gate["metrics"][0]["metric"] = "evil`| @octocat <img src=x> [x](http://e.invalid)"
    gate["summary"]["labels"]["candidate"] = "@everyone\n# heading"
    text = render_gate_markdown(gate, title="t")
    assert "<img" not in text
    assert "@octocat" not in text.replace("`", "")
    assert "\n# heading" not in text
    assert "](http" not in text.replace("`", "")


def test_malformed_gate_json_is_refused() -> None:
    with pytest.raises(ValueError):
        render_gate_markdown({"metrics": "nope"}, title="t")
    with pytest.raises(ValueError):
        render_gate_markdown({"summary": {}, "metrics": [{"metric": 3}]}, title="t")


def test_the_body_is_capped_below_the_github_limit() -> None:
    gate = _gate("v3_fluent_regression_vs_v1_baseline.json")
    gate["warnings"] = ["w" * 1000] * 200
    assert len(render_gate_markdown(gate, title="t")) <= 60_000


class FakeGitHub:
    """Issue comments for one repository, served like the REST API."""

    def __init__(self, head_sha: str = "abc123") -> None:
        self.head_sha = head_sha
        self.comments: list[dict[str, Any]] = []
        self.auth: list[str] = []
        self.next_id = 1
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(
                self, status: int, payload: Any, headers: dict[str, str] | None = None
            ) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self) -> dict[str, Any]:
                return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

            def do_GET(self) -> None:
                fake.auth.append(self.headers.get("Authorization", ""))
                url = urlsplit(self.path)
                if url.path == "/repos/o/r/pulls/7":
                    self._reply(200, {"number": 7, "head": {"sha": fake.head_sha}})
                    return
                if url.path == "/repos/o/r/issues/7/comments":
                    page = int(parse_qs(url.query).get("page", ["1"])[0])
                    chunk = fake.comments[(page - 1) * 2 : page * 2]  # two per page
                    headers = {}
                    if page * 2 < len(fake.comments):
                        headers["Link"] = (
                            f'<{fake.url}/repos/o/r/issues/7/comments?per_page=100&page={page + 1}>; rel="next"'
                        )
                    self._reply(200, chunk, headers)
                    return
                self._reply(404, {"message": "Not Found"})

            def do_POST(self) -> None:
                body = self._body()
                comment = {
                    "id": fake.next_id,
                    "body": body["body"],
                    "user": {"login": "github-actions[bot]", "type": "Bot"},
                }
                fake.next_id += 1
                fake.comments.append(comment)
                self._reply(201, comment)

            def do_PATCH(self) -> None:
                comment_id = int(self.path.rsplit("/", 1)[1])
                for c in fake.comments:
                    if c["id"] == comment_id:
                        c["body"] = self._body()["body"]
                        self._reply(200, c)
                        return
                self._reply(404, {"message": "Not Found"})

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def github() -> Iterator[FakeGitHub]:
    fake = FakeGitHub()
    fake.thread.start()
    yield fake
    fake.server.shutdown()
    fake.server.server_close()


def _human(text: str) -> dict[str, Any]:
    return {"id": 900, "body": text, "user": {"login": "someone", "type": "User"}}


def test_first_run_creates_the_comment_and_later_runs_update_it(github: FakeGitHub) -> None:
    api = GitHubAPI("dummy-token", api_url=github.url)
    github.comments = [_human("looks good"), _human(f"{MARKER} quoting the bot")]
    assert (
        upsert_sticky_comment(api, "o/r", 7, f"{MARKER}\nfirst", expected_head_sha="abc123")
        == "created"
    )
    assert (
        upsert_sticky_comment(api, "o/r", 7, f"{MARKER}\nsecond", expected_head_sha="abc123")
        == "updated"
    )
    bots = [c for c in github.comments if c["user"]["type"] == "Bot"]
    assert [c["body"] for c in bots] == [f"{MARKER}\nsecond"]
    assert (
        github.comments[1]["body"] == f"{MARKER} quoting the bot"
    )  # a person's comment is never edited
    assert all(a == "Bearer dummy-token" for a in github.auth)


def test_pagination_finds_the_bot_comment_on_a_later_page(github: FakeGitHub) -> None:
    api = GitHubAPI("t", api_url=github.url)
    github.comments = [_human(f"c{i}") for i in range(5)]
    upsert_sticky_comment(api, "o/r", 7, f"{MARKER}\none")
    github.comments = [_human(f"c{i}") for i in range(5)] + github.comments[5:]
    assert upsert_sticky_comment(api, "o/r", 7, f"{MARKER}\ntwo") == "updated"
    assert sum(1 for c in github.comments if c["user"]["type"] == "Bot") == 1


def test_a_comment_for_an_outdated_head_is_refused(github: FakeGitHub) -> None:
    api = GitHubAPI("t", api_url=github.url)
    with pytest.raises(HeadMismatch):
        upsert_sticky_comment(api, "o/r", 7, f"{MARKER}\nx", expected_head_sha="0ld")
    assert github.comments == []


def test_bad_repository_names_and_api_errors_are_reported(github: FakeGitHub) -> None:
    api = GitHubAPI("t", api_url=github.url)
    with pytest.raises(ValueError):
        upsert_sticky_comment(api, "o/r/../../x", 7, MARKER)
    with pytest.raises(GitHubError, match="404"):
        upsert_sticky_comment(api, "o/other", 7, MARKER, expected_head_sha="abc123")
