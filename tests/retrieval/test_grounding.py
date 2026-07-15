"""Grounding module — pure helpers + SQL + orchestration."""

from __future__ import annotations

import pytest

from engine.retrieval.grounding import (
    GroundingBundle,
    GroundingCandidate,
    _detect_bare_ids,
    _extract_tokens,
)

# ---- Token extraction ----------------------------------------------------

def test_extract_tokens_drops_stopwords():
    tokens = _extract_tokens("show me the auth refactor")
    assert "show" not in tokens
    assert "the" not in tokens
    assert "auth" in tokens
    assert "refactor" in tokens


def test_extract_tokens_lowercases():
    tokens = _extract_tokens("OAuth Migration")
    assert "oauth" in tokens
    assert "migration" in tokens


def test_extract_tokens_min_length_two():
    tokens = _extract_tokens("a b cd")
    assert "a" not in tokens
    assert "cd" in tokens


def test_extract_tokens_empty_query():
    assert _extract_tokens("") == []


def test_extract_tokens_drops_retrieval_noise():
    tokens = _extract_tokens("list the most recent commits")
    assert "list" not in tokens
    assert "commits" in tokens


def test_extract_tokens_preserves_file_path():
    tokens = _extract_tokens("prs about auth.py")
    assert "auth.py" in tokens


# ---- Bare ID detection --------------------------------------------------

def test_detect_bare_ids_linear_jira():
    ids = _detect_bare_ids("show PRs that closed ABC-123 and XYZ-9")
    assert ("ticket", "ABC-123") in ids
    assert ("ticket", "XYZ-9") in ids


def test_detect_bare_ids_github_pr():
    ids = _detect_bare_ids("review PR #49 in the backend")
    assert ("pr", "49") in ids


def test_detect_bare_ids_github_pr_conversational_forms():
    """Live-traced 2026-05-18: the original `#`-only regex missed
    natural English forms like 'Why was PR 328 created' — grounding
    fell back to fuzzy match on the repo slug and the PR's body
    chunks never made the prefanout. The broadened regex catches
    every common phrasing: `PR 328`, `pr 328`, `PR#328`, `PR-328`."""
    for query, expected in [
        ("Why was PR 328 in prbe-knowledge created", "328"),
        ("look up pr 49 in the backend", "49"),
        ("see PR#100 for context", "100"),
        ("PR-7 is the right reference", "7"),
        ("check pr.42 first", "42"),
    ]:
        ids = _detect_bare_ids(query)
        assert ("pr", expected) in ids, (
            f"failed to detect ('pr', {expected!r}) from {query!r}; got {ids}"
        )


def test_detect_bare_ids_issue_alias():
    """Issues share the PR namespace + phrasing; we tag both as `pr`
    so downstream grounding looks up whichever canonical_id form
    (pr: or issue:) the customer's graph has."""
    for query, expected in [
        ("Why was issue 77 raised", "77"),
        ("issue#42 status please", "42"),
    ]:
        ids = _detect_bare_ids(query)
        assert ("pr", expected) in ids, (
            f"failed to detect ('pr', {expected!r}) from {query!r}; got {ids}"
        )


def test_detect_bare_ids_no_false_positives_on_bare_numbers():
    """A bare number with no `#` / `PR` / `issue` prefix is NOT a
    PR — we must keep the regex tight enough that conversational
    numerics don't all get tagged as PRs."""
    for query in [
        "ship 3 features this week",
        "the last 7 commits were clean",
        "100x throughput on Cerebras",
    ]:
        ids = _detect_bare_ids(query)
        assert not any(kind == "pr" for kind, _ in ids), (
            f"false-positive PR detection in {query!r}: {ids}"
        )


def test_detect_bare_ids_git_sha_prefix():
    ids = _detect_bare_ids("revert 2d186dd please")
    assert any(kind == "commit_sha" for kind, _ in ids)


def test_detect_bare_ids_ignores_plain_numbers():
    ids = _detect_bare_ids("the last 3 commits")
    assert not any(kind == "pr" for kind, _ in ids)


def test_detect_bare_ids_empty():
    assert _detect_bare_ids("auth refactor status") == []


# ---- Dataclasses --------------------------------------------------------

def test_grounding_bundle_default_empty():
    b = GroundingBundle()
    assert b.candidates == []
    assert b.connected_sources == []
    assert b.bare_id_matches == []
    assert b.timing_ms == 0.0


def test_grounding_candidate_roundtrip():
    c = GroundingCandidate(
        entity_type="repo",
        canonical_id="prbe-backend",
        display_name="prbe-backend",
        last_seen_at=None,
        match_source="trgm",
    )
    assert c.canonical_id == "prbe-backend"
    assert c.match_source == "trgm"


# ---- Integration tests --------------------------------------------------

from engine.retrieval.grounding import (  # noqa: E402
    _connected_sources,
    _fuzzy_match_entities,
    _lookup_bare_id_matches,
    build_bundle,
)


@pytest.mark.integration
async def test_fuzzy_match_returns_seeded_entities(seeded_customer):
    candidates = await _fuzzy_match_entities(
        customer_id=seeded_customer.customer_id,
        tokens=["auth", "refactor"],
        per_type_cap=5, total_cap=20,
    )
    assert any(c.canonical_id == "auth-refactor" for c in candidates)


@pytest.mark.integration
async def test_fuzzy_match_caps_per_type(seeded_customer_many_repos):
    candidates = await _fuzzy_match_entities(
        customer_id=seeded_customer_many_repos.customer_id,
        tokens=["prbe"],
        per_type_cap=5, total_cap=20,
    )
    repo_count = sum(1 for c in candidates if c.entity_type == "repo")
    assert repo_count <= 5


@pytest.mark.integration
async def test_fuzzy_match_empty_tokens_returns_empty(seeded_customer):
    candidates = await _fuzzy_match_entities(
        customer_id=seeded_customer.customer_id,
        tokens=[], per_type_cap=5, total_cap=20,
    )
    assert candidates == []


@pytest.mark.integration
async def test_lookup_bare_id_matches_exact(seeded_customer):
    matches = await _lookup_bare_id_matches(
        customer_id=seeded_customer.customer_id,
        bare_ids=[("ticket", "ABC-123"), ("pr", "49")],
    )
    canon_ids = {(m.entity_type, m.canonical_id) for m in matches}
    assert ("ticket", "ABC-123") in canon_ids


@pytest.mark.integration
async def test_lookup_bare_id_misses_silently(seeded_customer):
    matches = await _lookup_bare_id_matches(
        customer_id=seeded_customer.customer_id,
        bare_ids=[("ticket", "NONEXISTENT-999")],
    )
    assert matches == []


@pytest.mark.integration
async def test_connected_sources(seeded_customer):
    sources = await _connected_sources(seeded_customer.customer_id)
    assert "github" in sources


@pytest.mark.integration
async def test_build_bundle_populates_all_fields(seeded_customer):
    bundle = await build_bundle(seeded_customer.customer_id, "auth refactor ABC-123")
    assert any(c.canonical_id == "auth-refactor" for c in bundle.candidates)
    assert any(m.canonical_id == "ABC-123" for m in bundle.bare_id_matches)
    assert "github" in bundle.connected_sources
    assert bundle.timing_ms >= 0


@pytest.mark.integration
async def test_build_bundle_empty_query(seeded_customer):
    bundle = await build_bundle(seeded_customer.customer_id, "")
    assert bundle.candidates == []
    assert bundle.bare_id_matches == []
    assert "github" in bundle.connected_sources


@pytest.mark.integration
async def test_build_bundle_partial_failure_returns_other_fields(seeded_customer, monkeypatch):
    async def boom(*_args, **_kwargs):
        raise RuntimeError("simulated graph_nodes failure")

    monkeypatch.setattr("engine.retrieval.grounding._fuzzy_match_entities", boom)
    bundle = await build_bundle(seeded_customer.customer_id, "auth ABC-123")
    assert bundle.candidates == []
    assert any(m.canonical_id == "ABC-123" for m in bundle.bare_id_matches)
    assert "github" in bundle.connected_sources


@pytest.mark.integration
async def test_build_bundle_total_failure_returns_empty(seeded_customer, monkeypatch):
    async def boom(*_args, **_kwargs):
        raise RuntimeError("simulated")

    monkeypatch.setattr("engine.retrieval.grounding._fuzzy_match_entities", boom)
    monkeypatch.setattr("engine.retrieval.grounding._lookup_bare_id_matches", boom)
    monkeypatch.setattr("engine.retrieval.grounding._connected_sources", boom)

    bundle = await build_bundle(seeded_customer.customer_id, "auth")
    assert bundle.candidates == []
    assert bundle.bare_id_matches == []
    assert bundle.connected_sources == []


# ---- Unit tests for fixes -----------------------------------------------

def test_match_source_fts_when_rel_is_exactly_half():
    """rel == 0.5 means trigram below floor but tsvector matched -> 'fts'.
    rel != 0.5 (above or below) means trigram contributed -> 'trgm'."""
    def classify(rel: float) -> str:
        return "trgm" if rel != 0.5 else "fts"

    assert classify(0.5) == "fts"
    assert classify(0.4) == "trgm"   # trigram only, below FTS floor
    assert classify(0.7) == "trgm"   # trigram dominant
    assert classify(0.0) == "trgm"   # technically impossible but doesn't matter


@pytest.mark.integration
async def test_build_bundle_handles_operator_chars_in_query(seeded_customer):
    """Operator chars like &, |, !, : in user input must not crash to_tsquery
    (regression: pre-fix `to_tsquery` would error on these; we use
    `plainto_tsquery` which strips operator semantics)."""
    bundle = await build_bundle(
        seeded_customer.customer_id, "auth&bug | refactor: status"
    )
    # candidates list is allowed to be empty; what matters is no crash
    # and that the bundle assembles. Use connected_sources as the
    # canary that the gather completed without all-failure.
    assert "github" in bundle.connected_sources


# ---- Doc-title channel (channel 4) --------------------------------------

from engine.retrieval.grounding import (  # noqa: E402
    _doc_id_to_entity_type,
    _fuzzy_match_document_titles,
)


def test_doc_id_to_entity_type_github_pr():
    assert _doc_id_to_entity_type("github:prbe-ai/prbe-knowledge:pr:340", "github") == "pr"


def test_doc_id_to_entity_type_github_issue():
    assert _doc_id_to_entity_type("github:prbe-ai/prbe-knowledge:issue:77", "github") == "ticket"


def test_doc_id_to_entity_type_github_commit():
    assert _doc_id_to_entity_type(
        "github:prbe-ai/prbe-knowledge:commit:abc1234", "github"
    ) == "commit_sha"


def test_doc_id_to_entity_type_linear():
    assert _doc_id_to_entity_type("linear:org:issue:abc", "linear") == "ticket"


def test_doc_id_to_entity_type_notion():
    assert _doc_id_to_entity_type("notion:page:abc", "notion") == "page"


def test_doc_id_to_entity_type_wiki_falls_to_document():
    assert _doc_id_to_entity_type("wiki:project:mg", "wiki") == "document"


def test_doc_id_to_entity_type_unknown_source_falls_to_document():
    assert _doc_id_to_entity_type("foo:bar:baz", "unknown_src") == "document"


def test_doc_id_to_entity_type_short_github_doc_id_falls_to_source_default():
    """When a GitHub doc_id has fewer than 3 colon-parts (malformed or
    repo-level), the structural parse falls through to the source_system
    map. `github` maps to `document` as a generic fallback."""
    assert _doc_id_to_entity_type("github:repo-only", "github") == "document"


@pytest.mark.integration
async def test_fuzzy_match_document_titles_returns_match_on_tsvector_hit(
    seeded_customer_with_docs,
):
    """The classic failing case: query 'multi-granola' must surface the
    Linear PRB-18 + Notion design rationale + wiki page as grounded
    candidates. Pre-channel-4, all three were Document nodes that
    grounding silently missed, leading to the phantom-entity / curation-
    lottery non-determinism this channel exists to fix."""
    candidates = await _fuzzy_match_document_titles(
        customer_id=seeded_customer_with_docs.customer_id,
        tokens=["multi-granola"],
    )
    canonical_ids = {c.canonical_id for c in candidates}
    assert "linear:org:issue:prb-18" in canonical_ids
    assert "notion:page:design-mg" in canonical_ids
    assert "wiki:project:multi_granola" in canonical_ids
    # All hits carry match_source for telemetry
    assert all(c.match_source == "doc_title" for c in candidates)
    # entity_type is derived from doc_id structure, not generic 'document'
    linear_cand = next(c for c in candidates if c.canonical_id == "linear:org:issue:prb-18")
    assert linear_cand.entity_type == "ticket"
    notion_cand = next(c for c in candidates if c.canonical_id == "notion:page:design-mg")
    assert notion_cand.entity_type == "page"


@pytest.mark.integration
async def test_fuzzy_match_document_titles_scopes_by_customer(
    seeded_customer_with_docs,
):
    """The seeded fixture includes a cross-tenant doc with the same
    'multi-granola' title under a DIFFERENT customer_id. The channel
    must NOT leak it across tenants. This is a defense-in-depth check
    on top of RLS — explicit WHERE customer_id = $1 in the SQL."""
    candidates = await _fuzzy_match_document_titles(
        customer_id=seeded_customer_with_docs.customer_id,
        tokens=["multi-granola"],
    )
    cross_tenant_id = "linear:org:issue:prb-cross"
    assert all(c.canonical_id != cross_tenant_id for c in candidates)


@pytest.mark.integration
async def test_fuzzy_match_document_titles_skips_soft_deleted(
    seeded_customer_with_docs,
):
    """Soft-deleted docs (valid_to IS NOT NULL) must not surface — they
    represent superseded versions whose titles may be stale or wrong.
    The SQL's `valid_to IS NULL` filter enforces this."""
    candidates = await _fuzzy_match_document_titles(
        customer_id=seeded_customer_with_docs.customer_id,
        tokens=["multi-granola"],
    )
    soft_deleted_id = "linear:org:issue:prb-old"
    assert all(c.canonical_id != soft_deleted_id for c in candidates)


@pytest.mark.integration
async def test_fuzzy_match_document_titles_empty_tokens_returns_empty(
    seeded_customer_with_docs,
):
    """Whitespace-only query produces no tokens; the channel short-
    circuits to [] without firing any SQL (no needless DB load on
    every empty request)."""
    candidates = await _fuzzy_match_document_titles(
        customer_id=seeded_customer_with_docs.customer_id,
        tokens=[],
    )
    assert candidates == []


@pytest.mark.integration
async def test_fuzzy_match_document_titles_fts_only_path_hits_body_preview(
    seeded_customer_with_docs,
):
    """When trgm similarity on title is below floor but tsvector FTS
    matches in body_preview, the doc still surfaces. The "Onboarding
    Runbook" doc has title="Onboarding Runbook" (zero trigram overlap
    with "kubernetes") but body_preview mentions kubernetes. The
    `idx_documents_fts_title_preview` GIN index covers both fields
    via `to_tsvector(title || ' ' || body_preview)`, so the FTS
    branch fires."""
    candidates = await _fuzzy_match_document_titles(
        customer_id=seeded_customer_with_docs.customer_id,
        tokens=["kubernetes"],
    )
    assert any(c.canonical_id == "notion:page:onboarding" for c in candidates)


@pytest.mark.integration
async def test_build_bundle_merges_doc_title_into_candidates(
    seeded_customer_with_docs,
):
    """Doc-title matches must merge into the bundle's `candidates` list
    (not a separate field), so downstream consumers see them through
    the existing API. Dedup-by-canonical_id keeps a doc from rendering
    twice if it happens to surface via both entity-fuzzy and
    doc-title channels."""
    bundle = await build_bundle(
        seeded_customer_with_docs.customer_id, "multi-granola plan"
    )
    canon_ids = {c.canonical_id for c in bundle.candidates}
    assert "linear:org:issue:prb-18" in canon_ids
    # No duplicate entries — canonical_id dedup
    seen: set[str] = set()
    for c in bundle.candidates:
        assert c.canonical_id not in seen, f"duplicate candidate: {c.canonical_id}"
        seen.add(c.canonical_id)
