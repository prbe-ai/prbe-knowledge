"""Unit tests for residualize_for_bm25 — strips identifier tokens and
identifier-frame descriptor words from a BM25 query when the router has
already extracted a stable identifier, so BM25 doesn't ts_rank_cd its
way through 10k unrelated chunks on a UUID-by-id query.
"""

from __future__ import annotations

from services.retrieval.retrievers.bm25 import residualize_for_bm25


def test_no_identifiers_returns_query_unchanged() -> None:
    # Without an extracted stable identifier, descriptor tokens like
    # "session" can be the actual topic — leave the query alone so BM25
    # keeps full recall on plain keyword searches.
    assert residualize_for_bm25("session timeout bug", []) == "session timeout bug"


def test_no_identifiers_empty_query_returns_none() -> None:
    assert residualize_for_bm25("", []) is None


def test_pure_identifier_query_returns_none() -> None:
    # "agent session <uuid>" — every token is either a descriptor or a
    # UUID hex part. id_lookup pins the doc; BM25 has no recall to add.
    result = residualize_for_bm25(
        "agent session 3c325e11-2008-46a9-83f7-fc40d11eaf82",
        ["3c325e11-2008-46a9-83f7-fc40d11eaf82"],
    )
    assert result is None


def test_identifier_plus_topical_returns_topical_residual() -> None:
    # id_lookup pins the session, but auth/refactor are real topical
    # signal. BM25 should still run, on just those tokens.
    result = residualize_for_bm25(
        "3c325e11-2008-46a9-83f7-fc40d11eaf82 auth refactor",
        ["3c325e11-2008-46a9-83f7-fc40d11eaf82"],
    )
    assert result is not None
    tokens = result.split()
    assert "auth" in tokens
    assert "refactor" in tokens
    # UUID hex parts must be stripped — they're already the id_lookup key.
    assert "3c325e11" not in tokens
    assert "fc40d11eaf82" not in tokens


def test_ticket_descriptor_stripped_with_ticket_id() -> None:
    result = residualize_for_bm25(
        "ticket PRB-17 enrichment workspace_prefs",
        ["PRB-17"],
    )
    assert result is not None
    tokens = result.split()
    assert "enrichment" in tokens
    assert "workspace_prefs" in tokens
    assert "ticket" not in tokens
    assert "PRB" not in tokens
    assert "17" not in tokens


def test_pr_descriptor_stripped_with_pr_ref() -> None:
    result = residualize_for_bm25(
        "pr prbe-backend#49 deploy timeout",
        ["prbe-backend#49"],
    )
    assert result is not None
    tokens = result.split()
    assert "deploy" in tokens
    assert "timeout" in tokens
    assert "pr" not in tokens
    # Repo slug parts are part of the canonical_id and get stripped too.
    assert "prbe" not in tokens
    assert "backend" not in tokens


def test_descriptor_match_is_case_insensitive() -> None:
    # Descriptor strip should fire on uppercase too — users frequently
    # paste UUIDs with the surrounding "AGENT SESSION" verbatim.
    result = residualize_for_bm25(
        "AGENT SESSION 3c325e11-2008-46a9-83f7-fc40d11eaf82",
        ["3c325e11-2008-46a9-83f7-fc40d11eaf82"],
    )
    assert result is None


def test_identifier_token_strip_is_case_insensitive() -> None:
    # Canonical_id uppercase, query lowercase — both should resolve to
    # the same lowered tokens in the stop set.
    result = residualize_for_bm25(
        "PRB-17 enrichment",
        ["prb-17"],
    )
    assert result is not None
    assert result == "enrichment"


def test_short_tokens_dropped() -> None:
    # Single-char tokens are ignored (matches _build_or_tsquery_string's
    # >=2-char threshold), and the UUID's first hex group is stripped as
    # an identifier token; only "auth" survives.
    result = residualize_for_bm25(
        "a 3c325e11 auth",
        ["3c325e11-2008-46a9-83f7-fc40d11eaf82"],
    )
    assert result == "auth"


def test_empty_query_with_identifiers_returns_none() -> None:
    assert (
        residualize_for_bm25("", ["3c325e11-2008-46a9-83f7-fc40d11eaf82"])
        is None
    )


def test_bare_descriptor_without_identifier_keeps_topic() -> None:
    # No identifier in routed entities — "session timeout" must remain
    # so BM25 ranks docs about session timeouts. Regression guard: the
    # descriptor stoplist is identifier-conditional, not unconditional.
    result = residualize_for_bm25("session timeout", [])
    assert result == "session timeout"
