# GitHub crawler eval fixtures

This directory holds the curated test corpus for **Lane D**'s GitHub
crawler. Lane D is the first concrete `BootstrapAgent` subclass; it
walks GitHub repos via the synthesis-side `GitHubAPIClient` and emits
wiki pages via the link-extracting writer hook (Lane B).

The corpus is **pre-built** before Lane D ships so the snapshot tests
are a quality gate from minute one. Until Lane D lands, the harness in
`test_github_crawler_snapshots.py` skips cleanly with a clear marker.

## How to read

- `fixtures/` — sample GitHub API JSON, organized as the API responses
  would land. Each fixture file matches the shape `GitHubAPIClient`
  yields (see `services/synthesis/api_clients/github.py`):
    - `repos.json` — `list_installation_repos` response
    - `<owner>/<repo>/repo.json` — `get_repo`
    - `<owner>/<repo>/pulls.json` — `list_pulls`
    - `<owner>/<repo>/issues.json` — `list_issues` (PRs filtered out)
    - `<owner>/<repo>/commits.json` — `list_commits`
    - `<owner>/<repo>/pull_reviews/<pr>.json` — `list_pull_reviews`
- `snapshots/pages/` — expected wiki page output (markdown +
  frontmatter), one file per page.
- `snapshots/wiki_links.json` — expected `wiki_links` rows produced by
  the link extractor running on every snapshot page.

## Universe

A mini "Probe org" that exercises every code path:

- **2 repos**: `prbe-ai/prbe-knowledge` (medium activity) and
  `prbe-ai/prbe-backend` (light: 1 PR + 1 issue).
- **3-4 personas**: `richard`, `maison`, `janedoe` (plus an unmapped
  `dependabot` commit for the null-author path).
- **PRs covering**:
  - `#42` — substantive architectural decision (drives a `decision/`
    page).
  - `#43` — bug fix (touches `service_card/auth`, doesn't create a new
    page).
  - `#44` — typo fix (noise; should be `skip_events()`'d).
  - `#45` — multi-reviewer (link-extraction across reviewer person
    pages).
  - `#46` — cross-references PR #42 and issue #18 (typed-link path).
  - `#41` — dependency bump (noise).
- **Issues covering**:
  - `#18` — decision driver (closed by PR #42).
  - `#19` — runbook-worthy ("how do we recover from X").
  - `#20` — closed-as-not-planned (noise).
  - `#21` — feature discussion.
- **Commits**: substantive (`feat(auth)…`, `decision: …`) plus noise
  (typo fix, dep bumps, dependabot).
- **PR reviews**: APPROVED / CHANGES_REQUESTED / COMMENTED states with
  comments that hint at decisions or trade-offs.

## How to add a new scenario

1. Drop new JSON into `fixtures/<owner>/<repo>/`. Match the shapes
   that `GitHubAPIClient` yields — see the existing files for
   reference.
2. Hand-write the expected wiki page(s) in `snapshots/pages/`. File
   name is `<wiki_type>__<slug>.md`. Include both YAML frontmatter and
   the markdown body — the link extractor reads both.
3. Append expected link tuples to `snapshots/wiki_links.json`. The
   extractor produces one row per `(dst_wiki_type, dst_slug,
   link_type, link_source)` combination after dedup; mirror that
   shape.
4. Add a test in `test_github_crawler_snapshots.py` that runs the
   scenario and asserts the diff is within epsilon.

## Edit-distance epsilon

Snapshots use a relaxed match because the wording of LLM output drifts
between runs. The harness uses `_normalize_for_compare(text)` (collapses
whitespace, lowercases) and accepts up to 15% character edit distance
per page (`_EDIT_DISTANCE_EPSILON = 0.15`).

The `wiki_links` snapshot is checked **exactly** — link extraction is
deterministic, so any drift is a parser regression rather than LLM
slop. The expected rows in `snapshots/wiki_links.json` were validated
by running `services.synthesis.wiki_links.extract_links` against every
page in `snapshots/pages/` (with frontmatter pre-parsed by `pyyaml`)
during fixture authoring; the set diff was zero before commit.

## Running

```bash
uv run pytest tests/synthesis/evals/github/ -v
```

Today (Lane D not landed), every test skips with the message
`Lane D (GitHub crawler) not yet implemented; snapshots reserved for
that PR.` Once `services.synthesis.crawlers.github.GitHubCrawlerAgent`
exists, the import succeeds, the `pytestmark` skip falls away, and the
tests run.

## Plug-in point for Lane D

`test_github_crawler_snapshots.py::_run_crawler_for_repo` is the seam.
Today it raises `NotImplementedError`. Lane D's PR replaces the body
to:

1. Construct a `GitHubAPIClient` with a static `"fixture"` bearer and
   the test's `httpx.AsyncClient` (respx routes intercept all traffic
   to `https://api.github.com/...`).
2. Instantiate `GitHubCrawlerAgent` with the fake client + an
   in-memory wiki state.
3. Run the crawler against the given `full_name`.
4. Return `{"pages": {...}, "wiki_links": [...],
   "skipped_pr_numbers": [...], "skipped_issue_numbers": [...],
   "skipped_commit_shas": [...]}`.

Total fixture + snapshot footprint: under 200 KB.
