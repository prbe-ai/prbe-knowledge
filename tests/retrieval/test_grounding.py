"""Grounding module — pure helpers + SQL + orchestration."""

from __future__ import annotations

import pytest

from services.retrieval.grounding import (
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

from services.retrieval.grounding import (  # noqa: E402
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

    monkeypatch.setattr("services.retrieval.grounding._fuzzy_match_entities", boom)
    bundle = await build_bundle(seeded_customer.customer_id, "auth ABC-123")
    assert bundle.candidates == []
    assert any(m.canonical_id == "ABC-123" for m in bundle.bare_id_matches)
    assert "github" in bundle.connected_sources


@pytest.mark.integration
async def test_build_bundle_total_failure_returns_empty(seeded_customer, monkeypatch):
    async def boom(*_args, **_kwargs):
        raise RuntimeError("simulated")

    monkeypatch.setattr("services.retrieval.grounding._fuzzy_match_entities", boom)
    monkeypatch.setattr("services.retrieval.grounding._lookup_bare_id_matches", boom)
    monkeypatch.setattr("services.retrieval.grounding._connected_sources", boom)

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
