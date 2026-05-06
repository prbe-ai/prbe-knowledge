"""Snapshot tests for the GitHub crawler (Lane D).

Lane D is not yet implemented. These tests are scaffolded so they:

  - SKIP cleanly today with a clear marker that Lane D hasn't landed
  - Run automatically once `services.synthesis.crawlers.github` exists

When Lane D ships, this file becomes a quality gate. Each fixture
scenario must produce wiki output within edit-distance epsilon of the
snapshot.

Fixtures (under ``fixtures/``) are curated GitHub API JSON responses
mirroring the shapes that ``GitHubAPIClient`` yields. Snapshots (under
``snapshots/``) are the expected wiki page bodies + ``wiki_links`` rows
that Lane D should emit when it walks those fixtures.

The mocking strategy mirrors ``tests/synthesis/test_github_api_client.py``
- ``respx`` routes intercept httpx traffic and serve the canned JSON.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

try:
    from services.synthesis.crawlers.github import (  # type: ignore[import-not-found]
        GitHubCrawlerAgent,
    )

    LANE_D_LANDED = True
except ImportError:
    GitHubCrawlerAgent = None  # type: ignore[assignment,misc]
    LANE_D_LANDED = False


pytestmark = pytest.mark.skipif(
    not LANE_D_LANDED,
    reason="Lane D (GitHub crawler) not yet implemented; snapshots reserved for that PR.",
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_FIXTURES = _HERE / "fixtures"
_SNAPSHOTS = _HERE / "snapshots"
_SNAPSHOT_PAGES = _SNAPSHOTS / "pages"
_SNAPSHOT_LINKS = _SNAPSHOTS / "wiki_links.json"

_GITHUB_API = "https://api.github.com"

# Edit-distance budget for relaxed page comparison. LLM output drifts;
# 15% per page is the soft ceiling. The wiki_links snapshot is checked
# exactly because link extraction is deterministic.
_EDIT_DISTANCE_EPSILON = 0.15


# ---------------------------------------------------------------------------
# Fixture + snapshot loading
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_repos_index() -> Any:
    return _read_json(_FIXTURES / "repos.json")


def _load_repo_fixtures(owner: str, repo: str) -> dict[str, Any]:
    """Load every fixture file for a single repo into a dict.

    Keys:
      - ``repo``: get_repo response
      - ``pulls``: list_pulls response
      - ``issues``: list_issues response
      - ``commits``: list_commits response
      - ``pull_reviews``: dict mapping PR number -> list_pull_reviews response
    """
    base = _FIXTURES / owner / repo
    pull_reviews_dir = base / "pull_reviews"
    pull_reviews: dict[int, Any] = {}
    if pull_reviews_dir.exists():
        for review_file in sorted(pull_reviews_dir.glob("*.json")):
            pull_reviews[int(review_file.stem)] = _read_json(review_file)
    return {
        "repo": _read_json(base / "repo.json"),
        "pulls": _read_json(base / "pulls.json"),
        "issues": _read_json(base / "issues.json"),
        "commits": _read_json(base / "commits.json"),
        "pull_reviews": pull_reviews,
    }


def _load_snapshot_page(name: str) -> str:
    """Read a snapshot page file by its on-disk name (without `.md`)."""
    return (_SNAPSHOT_PAGES / f"{name}.md").read_text()


def _load_expected_links() -> list[dict[str, Any]]:
    payload = _read_json(_SNAPSHOT_LINKS)
    if isinstance(payload, dict) and "links" in payload:
        return payload["links"]
    raise AssertionError(f"{_SNAPSHOT_LINKS} must be an object with a 'links' array.")


# ---------------------------------------------------------------------------
# respx wiring
# ---------------------------------------------------------------------------


def _install_routes(router: respx.Router) -> None:
    """Wire respx routes for every fixture under ``fixtures/``.

    The crawler's ``GitHubAPIClient`` will hit:
      - GET /installation/repositories
      - GET /repos/{owner}/{repo}
      - GET /repos/{owner}/{repo}/pulls
      - GET /repos/{owner}/{repo}/issues
      - GET /repos/{owner}/{repo}/commits
      - GET /repos/{owner}/{repo}/pulls/{n}/reviews

    Each route returns the canned JSON. Pagination headers are omitted
    because each fixture is a single page; if a future scenario needs
    multi-page pagination, add ``Link`` headers and a follow-up route.
    """
    repos_payload = _load_repos_index()
    router.get(f"{_GITHUB_API}/installation/repositories").mock(
        return_value=httpx.Response(200, json=repos_payload)
    )

    for repo_meta in repos_payload["repositories"]:
        full_name = repo_meta["full_name"]
        owner, repo = full_name.split("/", 1)
        bundle = _load_repo_fixtures(owner, repo)

        router.get(f"{_GITHUB_API}/repos/{full_name}").mock(
            return_value=httpx.Response(200, json=bundle["repo"])
        )
        router.get(re.compile(rf"^{_GITHUB_API}/repos/{full_name}/pulls(\?.*)?$")).mock(
            return_value=httpx.Response(200, json=bundle["pulls"])
        )
        router.get(re.compile(rf"^{_GITHUB_API}/repos/{full_name}/issues(\?.*)?$")).mock(
            return_value=httpx.Response(200, json=bundle["issues"])
        )
        router.get(re.compile(rf"^{_GITHUB_API}/repos/{full_name}/commits(\?.*)?$")).mock(
            return_value=httpx.Response(200, json=bundle["commits"])
        )
        for pr_number, review_payload in bundle["pull_reviews"].items():
            router.get(
                re.compile(rf"^{_GITHUB_API}/repos/{full_name}/pulls/{pr_number}/reviews(\?.*)?$")
            ).mock(return_value=httpx.Response(200, json=review_payload))


# ---------------------------------------------------------------------------
# Pytest fixtures (live behind the LANE_D_LANDED skip)
# ---------------------------------------------------------------------------


@pytest.fixture
async def github_http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def github_routes() -> AsyncIterator[respx.Router]:
    with respx.mock(assert_all_called=False) as router:
        _install_routes(router)
        yield router


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _normalize_for_compare(text: str) -> str:
    """Collapse whitespace + lowercase for relaxed page comparison.

    Markdown bodies drift between LLM runs. We strip trailing whitespace,
    collapse runs of whitespace to a single space, and lowercase. The
    edit-distance budget then absorbs the remaining slop.
    """
    stripped = text.strip().lower()
    return re.sub(r"\s+", " ", stripped)


def _edit_distance_ratio(a: str, b: str) -> float:
    """Levenshtein-distance / max(len) ratio. 0.0 = identical, 1.0 = total mismatch.

    Stdlib only — no python-Levenshtein dependency. O(len(a)*len(b)) but the
    snapshot pages are short (< 2KB) so this is fine.
    """
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m] / max(n, m)


def _assert_page_within_epsilon(actual: str, snapshot_name: str) -> None:
    """Compare ``actual`` against ``snapshots/pages/{snapshot_name}.md``.

    Both sides are normalized; the edit-distance ratio must be <=
    ``_EDIT_DISTANCE_EPSILON``. On mismatch, surface both bodies in the
    failure message so the snapshot can be hand-updated if the drift is
    intentional.
    """
    expected = _load_snapshot_page(snapshot_name)
    norm_actual = _normalize_for_compare(actual)
    norm_expected = _normalize_for_compare(expected)
    ratio = _edit_distance_ratio(norm_actual, norm_expected)
    assert ratio <= _EDIT_DISTANCE_EPSILON, (
        f"page {snapshot_name!r} drifted from snapshot "
        f"(edit ratio {ratio:.3f} > {_EDIT_DISTANCE_EPSILON}).\n"
        f"--- expected ---\n{expected}\n"
        f"--- actual ---\n{actual}\n"
    )


def _link_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        row["src_wiki_type"],
        row["src_slug"],
        row["dst_wiki_type"],
        row["dst_slug"],
        row["link_type"],
        row["link_source"],
    )


def _assert_links_match_exactly(actual: list[dict[str, Any]]) -> None:
    expected = _load_expected_links()
    actual_keys = {_link_key(r) for r in actual}
    expected_keys = {_link_key(r) for r in expected}
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    assert not missing and not extra, (
        f"wiki_links diverged from snapshot.\n"
        f"missing ({len(missing)}): {sorted(missing)}\n"
        f"extra   ({len(extra)}): {sorted(extra)}"
    )


# ---------------------------------------------------------------------------
# Lane D entry point — to be replaced when Lane D lands
# ---------------------------------------------------------------------------


async def _run_crawler_for_repo(
    http: httpx.AsyncClient,
    full_name: str,
) -> dict[str, Any]:
    """Run the GitHub crawler against the mocked fixtures for one repo.

    Returns ``{"pages": {snapshot_name: body_markdown, ...},
               "wiki_links": [link_row, ...],
               "skipped_pr_numbers": [...]}``.

    Implementation lives in Lane D. This stub will be filled in when
    ``GitHubCrawlerAgent`` exposes a ``run_for_repo()`` (or equivalent)
    that returns staged ``PageCreate`` / ``PageUpdate`` objects + the
    extracted ``wiki_links`` rows. Until then, the module-level
    ``pytestmark`` skips every test.
    """
    raise NotImplementedError(
        "Lane D plug-in point: invoke GitHubCrawlerAgent here, returning "
        "staged page bodies + extracted wiki_links rows for assertion."
    )


# ---------------------------------------------------------------------------
# Tests — one per fixture scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decision_page_extracted_from_substantive_pr(
    github_http: httpx.AsyncClient,
    github_routes: respx.Router,
) -> None:
    """The clean architectural PR (#42) produces a `decision/` page matching
    snapshots/pages/decision__pgvector-over-pinecone.md.

    Exercises: link-graph extraction (frontmatter contributors + related
    vendors, markdown person/vendor mentions), recency-first ordering
    (PR #42 is closed before #43/#44), and the rationale + trade-off
    structure that the prompt asks the agent to produce.
    """
    result = await _run_crawler_for_repo(github_http, "prbe-ai/prbe-knowledge")
    _assert_page_within_epsilon(
        result["pages"]["decision__pgvector-over-pinecone"],
        "decision__pgvector-over-pinecone",
    )


@pytest.mark.asyncio
async def test_bug_fix_pr_updates_existing_service_card(
    github_http: httpx.AsyncClient,
    github_routes: respx.Router,
) -> None:
    """The auth-refresh bug fix (#43) updates the existing service_card/auth
    page rather than creating a new one. Verifies the agent recognizes
    "this is maintenance on a known service" and uses update_page."""
    result = await _run_crawler_for_repo(github_http, "prbe-ai/prbe-knowledge")
    _assert_page_within_epsilon(
        result["pages"]["service_card__auth"],
        "service_card__auth",
    )


@pytest.mark.asyncio
async def test_noise_prs_dont_create_pages(
    github_http: httpx.AsyncClient,
    github_routes: respx.Router,
) -> None:
    """Typo PRs (#44) and dependency bumps (#41) produce no wiki page.

    The agent should call skip_events() for them rather than burning a
    create_page call. We assert by checking the skipped-PR list and
    confirming no snapshot key matches their numbers.
    """
    result = await _run_crawler_for_repo(github_http, "prbe-ai/prbe-knowledge")
    assert 44 in result["skipped_pr_numbers"]
    assert 41 in result["skipped_pr_numbers"]


@pytest.mark.asyncio
async def test_multi_reviewer_pr_yields_person_pages(
    github_http: httpx.AsyncClient,
    github_routes: respx.Router,
) -> None:
    """PR #45 has two reviewers (maison + janedoe). Each review author
    should produce a person/ page (or update an existing one), and the
    page bodies should match the snapshots."""
    result = await _run_crawler_for_repo(github_http, "prbe-ai/prbe-knowledge")
    _assert_page_within_epsilon(result["pages"]["person__maison"], "person__maison")
    _assert_page_within_epsilon(result["pages"]["person__janedoe"], "person__janedoe")
    _assert_page_within_epsilon(result["pages"]["person__richard"], "person__richard")


@pytest.mark.asyncio
async def test_pr_with_cross_reference_creates_typed_links(
    github_http: httpx.AsyncClient,
    github_routes: respx.Router,
) -> None:
    """PR #46 mentions PR #42 and issue #18 in its body. The agent's
    feature/wiki-bootstrap page should link to the decision and the
    related issue using `[[type:slug]]` syntax, and the link extractor
    should produce the rows in snapshots/wiki_links.json."""
    result = await _run_crawler_for_repo(github_http, "prbe-ai/prbe-knowledge")
    _assert_page_within_epsilon(
        result["pages"]["feature__wiki-bootstrap"],
        "feature__wiki-bootstrap",
    )


@pytest.mark.asyncio
async def test_runbook_issue_produces_runbook_page(
    github_http: httpx.AsyncClient,
    github_routes: respx.Router,
) -> None:
    """Issue #19 is explicitly framed as a runbook. The agent should
    create a runbook/ page mirroring snapshot
    runbook__recover-stuck-ingestion-drain.md, preserving the numbered
    steps."""
    result = await _run_crawler_for_repo(github_http, "prbe-ai/prbe-knowledge")
    _assert_page_within_epsilon(
        result["pages"]["runbook__recover-stuck-ingestion-drain"],
        "runbook__recover-stuck-ingestion-drain",
    )


@pytest.mark.asyncio
async def test_closed_as_not_planned_issue_is_skipped(
    github_http: httpx.AsyncClient,
    github_routes: respx.Router,
) -> None:
    """Issue #20 (Redis cache question, closed wontfix) should not produce
    a wiki page. The agent should call skip_events() for it."""
    result = await _run_crawler_for_repo(github_http, "prbe-ai/prbe-knowledge")
    skipped = result.get("skipped_issue_numbers", [])
    assert 20 in skipped


@pytest.mark.asyncio
async def test_noise_commits_dont_create_pages(
    github_http: httpx.AsyncClient,
    github_routes: respx.Router,
) -> None:
    """The dependabot ruff bump and the typo-fix commit should be
    skipped. The substantive auth-middleware commit may either feed
    service_card/auth or be skipped (PR #43 already covers it); we only
    assert the noise ones get skipped."""
    result = await _run_crawler_for_repo(github_http, "prbe-ai/prbe-knowledge")
    skipped_shas = set(result.get("skipped_commit_shas", []))
    assert "0718293a4b5c6d7e8f9001020304a5b6c7d8e9f0" in skipped_shas  # ruff bump
    assert "b2c3d4e5f607182939a4b5c6d7e8f90010203040" in skipped_shas  # typo fix
    assert "e5f607182930a4b5c6d7e8f900102030405a6b7c" in skipped_shas  # httpx bump


@pytest.mark.asyncio
async def test_wiki_link_extractor_service_card(
    github_http: httpx.AsyncClient,
    github_routes: respx.Router,
) -> None:
    """PR #45 ships the wiki link extractor as a discrete service. The
    agent should produce service_card/wiki-link-extractor."""
    result = await _run_crawler_for_repo(github_http, "prbe-ai/prbe-knowledge")
    _assert_page_within_epsilon(
        result["pages"]["service_card__wiki-link-extractor"],
        "service_card__wiki-link-extractor",
    )


@pytest.mark.asyncio
async def test_wiki_links_match_snapshot_exactly(
    github_http: httpx.AsyncClient,
    github_routes: respx.Router,
) -> None:
    """The link extractor is deterministic. Once the crawler emits its
    pages, the wiki_links rows should exactly match the snapshot."""
    knowledge = await _run_crawler_for_repo(github_http, "prbe-ai/prbe-knowledge")
    backend = await _run_crawler_for_repo(github_http, "prbe-ai/prbe-backend")
    combined = list(knowledge["wiki_links"]) + list(backend["wiki_links"])
    _assert_links_match_exactly(combined)


@pytest.mark.asyncio
async def test_lighter_repo_still_contributes(
    github_http: httpx.AsyncClient,
    github_routes: respx.Router,
) -> None:
    """The smaller `prbe-backend` repo (1 PR + 1 issue) should still
    contribute to the cross-repo feature/wiki-bootstrap page. Verifies
    the crawler walks every accessible repo, not just the big one."""
    result = await _run_crawler_for_repo(github_http, "prbe-ai/prbe-backend")
    # Backend repo's only substantive PR (#12) reinforces the
    # wiki-bootstrap feature page; we don't assert a separate snapshot
    # because the page is shared.
    assert 12 not in result.get("skipped_pr_numbers", [])
