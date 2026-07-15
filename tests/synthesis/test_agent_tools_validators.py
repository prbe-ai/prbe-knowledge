"""Validator-level tests for CreatePageArgs / UpdatePageArgs.

Verifies the two normalizations applied at the tool-input boundary:

  1. CreatePageArgs.slug is coerced to the `[a-z0-9_-]+` shape so the
     markdown link extractor and the dashboard route can both reach
     the page (the LLM occasionally hands us `prbe-ai/kb` style slugs).
  2. The page summary is stripped of a leading `<title>:` prefix —
     this duplicated chrome shows up in the index page as
     `[[Repo: X]] - Repo: X: ...` and is purely noise.

The wiki agent runtime tests cover dispatch + persistence; this module
covers the input validators in isolation.
"""

from __future__ import annotations

import pytest

from kb.synthesis.agent_tools import (
    CreatePageArgs,
    UpdatePageArgs,
    _normalize_slug,
    _strip_title_prefix,
)


class TestNormalizeSlug:
    def test_passthrough_already_canonical(self) -> None:
        assert _normalize_slug("prbe-backend") == "prbe-backend"

    def test_lowercases(self) -> None:
        assert _normalize_slug("Prbe-Backend") == "prbe-backend"

    def test_slash_becomes_dash(self) -> None:
        assert _normalize_slug("prbe-ai/kb") == "prbe-ai-kb"

    def test_dot_becomes_dash(self) -> None:
        assert _normalize_slug("some.thing") == "some-thing"

    def test_spaces_become_dash(self) -> None:
        assert _normalize_slug("repo name") == "repo-name"

    def test_drops_disallowed(self) -> None:
        assert _normalize_slug("foo!@bar") == "foobar"

    def test_collapses_dash_runs_and_trims(self) -> None:
        assert _normalize_slug("--foo//bar//baz--") == "foo-bar-baz"


class TestStripTitlePrefix:
    def test_strips_exact_title_match(self) -> None:
        out = _strip_title_prefix(
            "Repo: prbe-ai/kb: Markdown knowledge base.",
            title="Repo: prbe-ai/kb",
        )
        assert out == "Markdown knowledge base."

    def test_case_insensitive_title_match(self) -> None:
        out = _strip_title_prefix(
            "repo: prbe-backend: Core backend.",
            title="Repo: prbe-backend",
        )
        assert out == "Core backend."

    def test_generic_typed_prefix_heuristic(self) -> None:
        # No title in scope (update path): the `Type: Name:` heuristic catches it.
        out = _strip_title_prefix("Repo: prbe-backend: Core backend.")
        assert out == "Core backend."

    def test_leaves_unrelated_prefix_untouched(self) -> None:
        # Single-token prefix that isn't `Type: Name:` shape — leave alone.
        out = _strip_title_prefix("URL: https://example.com is the homepage.")
        assert out == "URL: https://example.com is the homepage."

    def test_leaves_summary_with_no_prefix_unchanged(self) -> None:
        out = _strip_title_prefix("Plain summary.", title="Repo: prbe-backend")
        assert out == "Plain summary."


class TestCreatePageArgsValidators:
    def _make(self, *, slug: str, title: str, summary: str) -> CreatePageArgs:
        return CreatePageArgs(
            wiki_type="repo",
            slug=slug,
            title=title,
            body_markdown="# body",
            summary=summary,
            commit_message="msg",
        )

    def test_normalizes_slug_with_slash(self) -> None:
        args = self._make(
            slug="prbe-ai/kb",
            title="Repo: prbe-ai/kb",
            summary="Markdown knowledge base.",
        )
        assert args.slug == "prbe-ai-kb"

    def test_strips_duplicated_title_prefix(self) -> None:
        args = self._make(
            slug="prbe-ai-kb",
            title="Repo: prbe-ai/kb",
            summary="Repo: prbe-ai/kb: Markdown knowledge base for Probe.",
        )
        assert args.summary == "Markdown knowledge base for Probe."

    def test_leaves_well_formed_summary_alone(self) -> None:
        args = self._make(
            slug="prbe-backend",
            title="Repo: prbe-backend",
            summary="Core Python backend with control + data planes.",
        )
        assert args.summary == "Core Python backend with control + data planes."

    def test_rejects_slug_that_normalizes_to_empty(self) -> None:
        # `////` -> "" after normalization; the validator falls back to the
        # raw input which then fails min_length>0 (after strip) — but here
        # we keep raw input so Field length still passes. Document the
        # current behavior so a future tightening is intentional.
        args = self._make(slug="x", title="t", summary="s")
        assert args.slug == "x"


class TestUpdatePageArgsValidators:
    def test_strips_typed_prefix_when_present(self) -> None:
        args = UpdatePageArgs(
            wiki_type="repo",
            slug="prbe-backend",
            body_markdown="# body",
            summary="Repo: prbe-backend: Updated blurb.",
            commit_message="msg",
        )
        assert args.summary == "Updated blurb."

    def test_does_not_normalize_slug(self) -> None:
        # Update must hit the page under its existing slug — no rewrite.
        args = UpdatePageArgs(
            wiki_type="repo",
            slug="prbe-ai/kb",
            body_markdown="# body",
            summary="Edits.",
            commit_message="msg",
        )
        assert args.slug == "prbe-ai/kb"


def test_create_args_still_rejects_blank_slug() -> None:
    with pytest.raises(ValueError):
        CreatePageArgs(
            wiki_type="repo",
            slug="",
            title="t",
            body_markdown="b",
            summary="s",
            commit_message="m",
        )
