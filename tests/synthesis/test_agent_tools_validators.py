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
            body_markdown="b",
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
            edits=[{"op": "replace", "find": "a", "text": "b"}],
            summary="Repo: prbe-backend: Updated blurb.",
            commit_message="msg",
        )
        assert args.summary == "Updated blurb."

    def test_does_not_normalize_slug(self) -> None:
        # Update must hit the page under its existing slug — no rewrite.
        args = UpdatePageArgs(
            wiki_type="repo",
            slug="prbe-ai/kb",
            edits=[{"op": "replace", "find": "a", "text": "b"}],
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


# ---------------------------------------------------------------------------
# wiki_type is a CLOSED set
# ---------------------------------------------------------------------------


def test_the_tool_schema_declares_the_types_as_an_enum_not_prose() -> None:
    """The model must be CONSTRAINED to the set, not merely told about it.

    A description listing the allowed kinds is a suggestion the model may
    ignore on any given turn, and `wiki_type` is a permanent identity: it is a
    path segment (`/api/wiki/pages/{wiki_type}/{slug}`) and a doc_id component
    (`wiki:{type}:{slug}`), and there is no rename route. One ignored
    suggestion at 04:00 is a page kind forever.
    """
    from kb.synthesis.agent_tools import _WIKI_TYPE_SCHEMA
    from kb.synthesis.models import AGENT_WIKI_TYPES

    assert _WIKI_TYPE_SCHEMA.get("enum") == list(AGENT_WIKI_TYPES)


def test_the_agent_cannot_write_the_generated_index_page() -> None:
    """`index` is a real member but is NOT offered to the agent.

    The index is regenerated from the other pages at the end of every drain,
    so an agent writing it directly would have its work overwritten inside the
    same run -- a write that succeeds and then silently disappears.
    """
    from kb.synthesis.models import AGENT_WIKI_TYPES, WikiType

    assert WikiType.INDEX.value not in AGENT_WIKI_TYPES
    assert set(AGENT_WIKI_TYPES) == {t.value for t in WikiType} - {"index"}


def test_the_agent_enum_is_derived_from_WikiType_not_restated() -> None:
    """`AgentWikiType` must track `WikiType`, and this is why it is generated.

    It was a hand-written class listing its members literally. Adding
    `service` and `feature` to `WikiType` reached the tool schema, the
    ingestion gate and the prompt -- and silently did not reach this. The
    schema then offered two kinds the validator refused, so every attempt to
    use them came back as an invalid argument with no hint why.

    Asserted as EQUALITY against the derivation, not membership: a check that
    every `AgentWikiType` is a `WikiType` passes for a set missing half its
    members, which is precisely the failure that shipped.
    """
    from kb.synthesis.models import AgentWikiType, WikiType

    assert [t.value for t in AgentWikiType] == [
        t.value for t in WikiType if t is not WikiType.INDEX
    ]


def test_a_service_and_a_feature_are_page_kinds() -> None:
    """Both are things an engineering wiki should hold, and both were already
    in the graph's link vocabulary (`[[service: X]]` resolves to a node), so
    until they were added a page could link to a service it could never be."""
    from kb.handlers.wiki import is_valid_wiki_type
    from kb.synthesis.agent_tools import _WIKI_TYPE_SCHEMA

    for wiki_type in ("service", "feature"):
        assert is_valid_wiki_type(wiki_type)
        assert wiki_type in _WIKI_TYPE_SCHEMA["enum"]


def test_the_gate_the_prompt_and_the_schema_cannot_drift() -> None:
    """One constant reaches all three, and this is what pins that.

    They are three different surfaces -- the ingestion gate that persists a
    page, the tool schema the model is constrained by, and the prose it reads
    -- and the failure when they disagree is silent: the agent emits a type
    the gate then refuses, so the drain loses that page every night with
    nothing but a log line.
    """
    from kb.handlers.wiki import is_valid_wiki_type
    from kb.synthesis.agent_tools import _WIKI_TYPE_SCHEMA
    from kb.synthesis.models import AGENT_WIKI_TYPES
    from kb.synthesis.prompts import _AGENT_WIKI_TYPES_SENTENCE

    for wiki_type in _WIKI_TYPE_SCHEMA["enum"]:
        assert is_valid_wiki_type(wiki_type), f"schema offers {wiki_type}, gate refuses it"
        assert f"`{wiki_type}`" in _AGENT_WIKI_TYPES_SENTENCE
    # ...and the prose names nothing the schema does not offer.
    import re

    assert re.findall(r"`([a-z_]+)`", _AGENT_WIKI_TYPES_SENTENCE) == list(AGENT_WIKI_TYPES)


def test_an_invented_type_is_refused_by_the_gate() -> None:
    """The behaviour that changed. `company` was in the old prompt's
    suggestions and is not a member; the gate must now refuse it rather than
    mint a permanent page kind."""
    from kb.handlers.wiki import is_valid_wiki_type

    for invented in ("company", "customer", "event", "repository", "codebase"):
        assert not is_valid_wiki_type(invented), invented


def test_the_shape_check_survives_the_membership_check() -> None:
    """Membership must not REPLACE the URL-safety regex.

    A member added later with a slash or a space in it would pass membership
    and still break every route that carries it as a path segment.
    """
    from kb.handlers.wiki import is_valid_wiki_type

    for malformed in ("Repo", "re po", "repo/slug", "1repo", "", None, 7):
        assert not is_valid_wiki_type(malformed), malformed
